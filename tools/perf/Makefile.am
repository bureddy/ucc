#
# Copyright (c) 2021 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = ucc_perftest

ucc_perftest_SOURCES =      \
	ucc_perftest.cc         \
	ucc_pt_config.cc        \
	ucc_pt_comm.cc          \
	ucc_pt_benchmark.cc     \
	ucc_pt_coll.cc          \
	ucc_pt_bootstrap_mpi.cc

CXX=$(MPICXX)
LD=$(MPICXX)
ucc_perftest_CPPFLAGS=$(BASE_CPPFLAGS)
ucc_perftest_CXXFLAGS=-std=gnu++11 $(BASE_CXXFLAGS)
ucc_perftest_LDADD = $(UCC_TOP_BUILDDIR)/src/libucc.la
