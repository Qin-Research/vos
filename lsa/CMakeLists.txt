find_package(HDF5)

include_directories(/home/masa/research/opengm/src/external/MaxFlow-v3.02.src-patched/)
link_directories(/home/masa/research/opengm/build/src/external/)
add_executable(solve_lsa solve_lsa.cpp)
target_link_libraries(solve_lsa external-library-maxflow-shared ${HDF5_LIBRARIES})