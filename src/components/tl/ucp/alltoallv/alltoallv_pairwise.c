/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_mc.h"
#include "cuda_runtime.h"

#define DGX1 1

#if DGX1

/* local group. 3rd bit is same*/
#define IS_NVLINK_PROXY_SRC(_rank1, _rank2) \
        ((_rank1 != _rank2) && ((_rank1 & 0x4) == (_rank2 & 0x4)))

/* flip 3rd bit */
#define NVLINK_PROXY_TARGET(_rank) (_rank ^ 0x4)

#elif ZIONEX
/* local group, 2 nd bit is samae*/
#define IS_NVLINK_PROXY_SRC(_rank1, _rank2) \
        ((_rank1 != _rank2) && ((_rank1 & 0x4) == (_rank2 & 0x4)))

/*flip 2,3 rd bits*/
#define NVLINK_PROXY_TARGET(_rank) (_rank ^ 0x6)

#endif

#define IS_NVLINK_ACCEESSIBLE(_rank1, _rank2, _is_cube_mesh_nvlink) \
        (!_is_cube_mesh_nvlink || ((IS_NVLINK_PROXY_SRC(_rank1, _rank2)) || (NVLINK_PROXY_TARGET(_rank1) == _rank2)))

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)coll_task->args.src.info_v.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)coll_task->args.dst.info_v.buffer;
    ucc_memory_type_t  smem  = coll_task->args.src.info_v.mem_type;
    ucc_memory_type_t  rmem  = coll_task->args.dst.info_v.mem_type;
    ucc_rank_t         grank = team->rank;
    ucc_rank_t         gsize = team->size;
    int                polls = 0;
    uint32_t           to_send_post = gsize;
    uint32_t           to_recv_post = gsize;
    ucc_rank_t         peer;
    int                posts, nreqs;//, count_stride, displ_stride;
    size_t             rdt_size, sdt_size, data_size, data_displ, ipc_thresh;

    if (task->alltoall_intra.info) {
        to_send_post -= task->alltoall_intra.send_posted;
        to_recv_post -= task->alltoall_intra.recv_posted;
    }

    ipc_thresh = UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoallv_ipc_thresh;
    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    rdt_size = ucc_dt_size(coll_task->args.src.info_v.datatype);
    sdt_size = ucc_dt_size(coll_task->args.dst.info_v.datatype);
    while ((task->send_posted < gsize || task->recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->recv_posted < gsize) &&
               ((task->recv_posted - task->recv_completed) < nreqs)) {
            peer       = get_recv_peer(grank, gsize, task->recv_posted);
            data_size =
                ucc_coll_args_get_count(
                    &coll_task->args, coll_task->args.dst.info_v.counts, peer) *
                rdt_size;
            if (IS_RANK_LOCAL(team, peer) && data_size >= ipc_thresh && to_recv_post != gsize) {
                task->recv_posted++;
                task->recv_completed++;
                continue;
            }

            data_displ = ucc_coll_args_get_displacement(
                             &coll_task->args,
                             coll_task->args.dst.info_v.displacements, peer) *
                         rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nz((void *)(rbuf + data_displ),
                                             data_size, rmem, peer, team, task),
                          task, out);
           // tl_warn(UCC_TL_TEAM_LIB(team), "UCX recv posted [%d:%d]", team->rank, peer);

            polls = 0;
        }
        while ((task->send_posted < gsize) &&
               ((task->send_posted - task->send_completed) < nreqs)) {
            peer       = get_send_peer(grank, gsize, task->send_posted);
            data_size =
                ucc_coll_args_get_count(
                    &coll_task->args, coll_task->args.src.info_v.counts, peer) *
                sdt_size;
            if (IS_RANK_LOCAL(team, peer) && data_size >= ipc_thresh && to_send_post != gsize) {
                task->send_posted++;
                task->send_completed++;
                continue;
            }
            data_displ = ucc_coll_args_get_displacement(
                             &coll_task->args,
                             coll_task->args.src.info_v.displacements, peer) *
                         sdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nz((void *)(sbuf + data_displ),
                                             data_size, smem, peer, team, task),
                          task, out);
            // tl_warn(UCC_TL_TEAM_LIB(team), "UCX send posted [%d:%d]", team->rank, peer);
            polls = 0;
        }
    }
    if ((task->send_posted < gsize) || (task->recv_posted < gsize)) {
        return task->super.super.status;
    }
    task->super.super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoallv_pairwise_done", 0);
    }

    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoallv_pairwise_start",
                                     0);
    ucc_tl_ucp_alltoallv_pairwise_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucs_status_t ucc_tl_ucp_alltoallv_cuda_ipc_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t intra_rank_start = ucs_align_down(team->rank, INTRA_PPN);
    ucc_rank_t intra_rank_end   = ucs_min(intra_rank_start + INTRA_PPN, team->size) - 1;
    ucc_rank_t intra_rank       = team->rank-intra_rank_start;
    int     is_cube_mesh_nvlink = UCC_TL_UCP_TEAM_CTX(team)->cfg.cube_mesh_nvlink;
    ucc_status_t status;
    void *base_address;
    size_t alloc_length, sdt_size, rdt_size, ipc_thresh;
    int i, j, coll_id;
    mem_info_t *my_info;
    mem_info_t *peer_info;
    void *mapped_addr;
    size_t total_counts;
    char aa[512], bb[512];
    char *a = aa, *b =bb;

    ipc_thresh = UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoallv_ipc_thresh;
    coll_id = (task->tag % MAX_ALLTOALLV_CONCURRENT);
    peer_info = &team->a2av[NODE_GROUP_SIZE * coll_id];
    my_info = &peer_info[NODE_RANK(team)];

    rdt_size = ucc_dt_size(coll_task->args.dst.info_v.datatype);
    sdt_size = ucc_dt_size(coll_task->args.src.info_v.datatype);

    total_counts = ucc_coll_args_get_total_count(&coll_task->args, coll_task->args.src.info_v.counts, team->size);
    ucc_tl_ucp_get_alloc_info(coll_task->args.src.info_v.buffer, total_counts * sdt_size,  &base_address, &alloc_length);
    if (base_address != NULL) {
        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info->src.handle, base_address));
    }
    my_info->src.d_ptr  = base_address;
    my_info->src.size   = alloc_length;
    my_info->src.offset = coll_task->args.src.info_v.buffer - base_address;

    total_counts = ucc_coll_args_get_total_count(&coll_task->args, coll_task->args.dst.info_v.counts, team->size);
    ucc_tl_ucp_get_alloc_info(coll_task->args.dst.info_v.buffer, total_counts * rdt_size,  &base_address, &alloc_length);
    if (base_address != NULL) {
        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info->dst.handle, base_address));
    }
    my_info->dst.d_ptr  = base_address;
    my_info->dst.size   = alloc_length;
    my_info->dst.offset = coll_task->args.dst.info_v.buffer - base_address;

    a += sprintf(a, "[ %d : ", intra_rank);
    b += sprintf(b, "[ %d : ", intra_rank);
    for (i = intra_rank_start, j = 0; i <= intra_rank_end; i++, j++) {

        my_info->src.displ[j] =  ucc_coll_args_get_displacement(&coll_task->args,
                coll_task->args.src.info_v.displacements,i) * sdt_size;
        my_info->src.length[j]  = ucc_coll_args_get_count(&coll_task->args,
                            coll_task->args.src.info_v.counts, i) * sdt_size;
        a += sprintf(a, "  %5ld ", my_info->src.length[j]);
        my_info->dst.displ[j] =  ucc_coll_args_get_displacement(&coll_task->args,
                coll_task->args.dst.info_v.displacements,i) * rdt_size;
        my_info->dst.length[j]  = ucc_coll_args_get_count(&coll_task->args,
                            coll_task->args.dst.info_v.counts, i) * rdt_size;
        b += sprintf(b, "  %5ld ", my_info->dst.length[j]);
        my_info->ev_handle[j] = team->ipc_event_handle[coll_id][j];
        CUDACHECK(cudaEventRecord(team->event[coll_id][j], (cudaStream_t)coll_task->ee->ee_context));
    }

    tl_debug(UCC_TL_TEAM_LIB(team), "SEND LEN: %s", aa);
    tl_debug(UCC_TL_TEAM_LIB(team), "RECV LEN: %s", bb);

    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[0] = (task->tag + 1);

    for (j = 0; j < NODE_GROUP_SIZE; j++) {
        volatile mem_info_t *pi = peer_info;
        while (pi[j].seq_num[0] != (task->tag + 1));
    }
    for (i=intra_rank_start,j = 0 ; i <= intra_rank_end; i++, j++) {
        if (i != team->rank) {
            ucc_assert(j < INTRA_PPN);

            if (IS_NVLINK_ACCEESSIBLE(j, intra_rank, is_cube_mesh_nvlink) &&
                    ((peer_info[intra_rank].dst.length[j] > ipc_thresh) ||
                     (is_cube_mesh_nvlink && peer_info[NVLINK_PROXY_TARGET(intra_rank)].dst.length[j] > ipc_thresh))) {

                status = ucc_cuda_ipc_map_memhandle(peer_info[j].src.d_ptr, peer_info[j].src.size,
                        peer_info[j].src.handle, &mapped_addr,
                        UCC_TL_UCP_TEAM_CTX(team)->ipc_cache[j]);
                if (UCC_OK != status) {
                    ucc_error("ucc_cuda_ipc_map_memhandle failed");
                    return UCC_ERR_INVALID_PARAM;
                }
                task->alltoall_intra.peer_src_map_addr[j] = mapped_addr;
            }

            if (is_cube_mesh_nvlink && (NVLINK_PROXY_TARGET(intra_rank) == j)) {
                status = ucc_cuda_ipc_map_memhandle(peer_info[j].dst.d_ptr, peer_info[j].dst.size,
                        peer_info[j].dst.handle, &mapped_addr,
                        UCC_TL_UCP_TEAM_CTX(team)->ipc_cache[j]);
                if (UCC_OK != status) {
                    ucc_error("ucc_cuda_ipc_map_memhandle failed");
                    return UCC_ERR_INVALID_PARAM;
                }
                task->alltoall_intra.peer_dst_map_addr[j] = mapped_addr;
            }

        }

        if(i != team->rank) {
            if (team->ipc_event[coll_id][j] == (cudaEvent_t) NULL) {
                CUDACHECK(cudaIpcOpenEventHandle(&team->ipc_event[coll_id][j], peer_info[j].ev_handle[intra_rank]));
            }
        }
    }

    task->alltoall_intra.coll_id  = coll_id;
    task->alltoall_intra.info = peer_info;

    return UCC_OK;
}

#define IPC_GET_ALLTOALLV_SEND_BUF_INFO(_task, _addr, _size, _rank, _target) \
{                                                                               \
    mem_info_t *info = &((mem_info_t *)_task->alltoall_intra.info)[_rank]; \
    _addr = (ptrdiff_t) _task->alltoall_intra.peer_src_map_addr[_rank] + info->src.offset + info->src.displ[_target]; \
    _size = info->src.length[_target]; \
}

#define IPC_GET_ALLTOALLV_RECV_BUF_INFO(_task, _addr, _size, _rank, _target) \
{                                                                               \
    mem_info_t *info = &((mem_info_t *)_task->alltoall_intra.info)[_rank]; \
    _addr = (ptrdiff_t) _task->alltoall_intra.peer_dst_map_addr[_rank] + info->dst.offset + info->dst.displ[_target]; \
    _size = info->dst.length[_target];    \
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_early_triggered_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ptrdiff_t          rbuf     = (ptrdiff_t)coll_task->args.dst.info_v.buffer;
    ucc_rank_t intra_rank_start = ucs_align_down(team->rank, INTRA_PPN);
    ucc_rank_t intra_rank_end   = ucs_min(intra_rank_start + INTRA_PPN, team->size) - 1;
    ucc_rank_t intra_rank       = team->rank - intra_rank_start;
    int     is_cube_mesh_nvlink = UCC_TL_UCP_TEAM_CTX(team)->cfg.cube_mesh_nvlink;
    size_t   rdt_size, sdt_size, data_size, data_displ, ipc_thresh;
    int rank, i, j, peer;
    mem_info_t *peer_info, *my_info;
    ptrdiff_t src, dst;

    task->alltoall_intra.send_posted = 0;
    task->alltoall_intra.recv_posted = 0;
    if (!UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoall_use_ipc) {
        return UCC_OK;
    }
    ucc_tl_ucp_alltoallv_cuda_ipc_setup(coll_task);

    ipc_thresh = UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoallv_ipc_thresh;
    rdt_size = ucc_dt_size(coll_task->args.dst.info_v.datatype);
    sdt_size = ucc_dt_size(coll_task->args.src.info_v.datatype);

    /* Direct copy over NVLINK */
    for (j=0; j < INTRA_PPN; j++) {
        rank = team->rank + j;
        if (rank > intra_rank_end) {
            rank = intra_rank_start - 1 + ((rank - intra_rank_end) % INTRA_PPN);
        }
        peer = rank - intra_rank_start;
        peer_info = &((mem_info_t *)task->alltoall_intra.info)[peer];

        if (rank == team->rank) {
            src = (ptrdiff_t)coll_task->args.src.info_v.buffer +
                    + peer_info->src.displ[intra_rank];
        } else {
            src = (ptrdiff_t) task->alltoall_intra.peer_src_map_addr[peer] +
                    peer_info->src.offset + peer_info->src.displ[intra_rank];
        }

        data_size  = ucc_coll_args_get_count(&coll_task->args,
                            coll_task->args.dst.info_v.counts, rank) * rdt_size;
        if (data_size < ipc_thresh && rank != team->rank) {
            continue;
        }
        data_displ = ucc_coll_args_get_displacement(&coll_task->args,
                            coll_task->args.dst.info_v.displacements, rank)* rdt_size;

        if (data_size != 0) {
            if (rank == team->rank) {
                CUDACHECK(cudaMemcpyAsync((void *)(rbuf + data_displ), (void *)src, data_size, cudaMemcpyDeviceToDevice, (cudaStream_t)coll_task->ee->ee_context));
            } else if (IS_NVLINK_ACCEESSIBLE(peer, intra_rank, is_cube_mesh_nvlink)) {
                CUDACHECK(cudaStreamWaitEvent((cudaStream_t)coll_task->ee->ee_context,
                            team->ipc_event[task->alltoall_intra.coll_id][peer], 0));

                CUDACHECK(cudaMemcpyAsync((void *)(rbuf + data_displ), (void *)src, data_size, cudaMemcpyDeviceToDevice, (cudaStream_t)coll_task->ee->ee_context));

                if (!is_cube_mesh_nvlink) {
                    CUDACHECK(cudaEventRecord(team->ipc_event[task->alltoall_intra.coll_id][peer], (cudaStream_t)coll_task->ee->ee_context));
                }
            }
        }
    }

    /* Indirect copies over NVLINK */
    if (is_cube_mesh_nvlink) {
        int dst_peer, src_peer;
        size_t s_data_size, d_data_size;
        dst_peer = NVLINK_PROXY_TARGET(intra_rank);
        for (src_peer = 0; src_peer < INTRA_PPN; src_peer++) {
            if (IS_NVLINK_PROXY_SRC(src_peer, intra_rank)) {
                IPC_GET_ALLTOALLV_SEND_BUF_INFO(task, src, s_data_size, src_peer, dst_peer);
                IPC_GET_ALLTOALLV_RECV_BUF_INFO(task, dst, d_data_size, dst_peer, src_peer);
                ucc_assert(s_data_size == d_data_size);
                if (s_data_size != 0 && s_data_size >= ipc_thresh) {
                    CUDACHECK(cudaMemcpyAsync((void *)dst, (void *)src, s_data_size, cudaMemcpyDeviceToDevice, (cudaStream_t)coll_task->ee->ee_context));
                }
                CUDACHECK(cudaEventRecord(team->ipc_event[task->alltoall_intra.coll_id][src_peer], (cudaStream_t)coll_task->ee->ee_context));
            }
            CUDACHECK(cudaEventRecord(team->ipc_event[task->alltoall_intra.coll_id][dst_peer], (cudaStream_t)coll_task->ee->ee_context));
        }
    }

    peer_info = &team->a2av[NODE_GROUP_SIZE * task->alltoall_intra.coll_id];
    my_info = &peer_info[NODE_RANK(team)];


    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[1] = (task->tag + 1);

    for (j = 0; j < NODE_GROUP_SIZE; j++) {
        volatile mem_info_t *pi = peer_info;
        while (pi[j].seq_num[1] != (task->tag + 1));
    }

    for (i=intra_rank_start,j = 0 ; i <= intra_rank_end; i++, j++) {
        peer_info = &((mem_info_t *)task->alltoall_intra.info)[j];
        if (!is_cube_mesh_nvlink) {
            if (i != team->rank) {
                data_size  = ucc_coll_args_get_count(&coll_task->args,
                        coll_task->args.src.info_v.counts, i) * sdt_size;
                if (data_size != 0) {
                    CUDACHECK(cudaStreamWaitEvent((cudaStream_t)coll_task->ee->ee_context, team->event[task->alltoall_intra.coll_id][j], 0));
                }
            }
        } else {
            if (IS_NVLINK_ACCEESSIBLE(j, intra_rank, 1)) {
                CUDACHECK(cudaStreamWaitEvent((cudaStream_t)coll_task->ee->ee_context, team->event[task->alltoall_intra.coll_id][j], 0));
            }
        }

        if (j != intra_rank) {
            if (peer_info->src.length[j] >= ipc_thresh) task->alltoall_intra.send_posted++;
            if (peer_info->dst.length[j] >= ipc_thresh) task->alltoall_intra.recv_posted++;

        } else {
            task->alltoall_intra.send_posted++;
            task->alltoall_intra.recv_posted++;
        }


    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &task->super.args;

    task->super.post     = ucc_tl_ucp_alltoallv_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoallv_pairwise_progress;
    task->super.early_triggered_post  = ucc_tl_ucp_alltoallv_pairwise_early_triggered_post;

    task->n_polls = ucc_min(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->src.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                               team->size) *
                 ucc_dt_size(args->src.info_v.datatype)),
                args->src.info_v.mem_type);
        }

        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->dst.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                               team->size) *
                 ucc_dt_size(args->dst.info_v.datatype)),
                args->dst.info_v.mem_type);
        }
    }

    ucc_tl_ucp_task_reset(task);

    return UCC_OK;
}
