

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source} ../util/util.h)
    target_link_libraries(${name} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)
