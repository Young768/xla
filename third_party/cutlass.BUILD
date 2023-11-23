# Description:
# The cutlass API is a C++ header-only library that demonstrates how
# to use the cutlass C backend API.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "cutlass_header_files",
    srcs = glob([
        "include/**",
    ]),
)

cc_library(
    name = "cutlass",
    hdrs = [":cutlass_header_files"],
    strip_include_prefix = "/include",
)
