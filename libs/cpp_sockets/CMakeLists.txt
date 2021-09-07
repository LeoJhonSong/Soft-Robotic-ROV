find_package(OpenCV REQUIRED) # Transfer images
set (EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})

include_directories (include)

# Create library
add_library(cpp_sockets
    src/client.cpp
)
target_link_libraries(cpp_sockets ${EXTRA_LIBS})

# Library installation:
install (TARGETS cpp_sockets DESTINATION bin)
install (FILES include/client.hpp DESTINATION include)