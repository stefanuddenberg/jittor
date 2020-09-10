# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import sys
import shutil
from .compiler import *
from jittor_utils import run_cmd, get_version
from jittor.utils.misc import download_url_to_local


def search_file(dirs, name):
    for d in dirs:
        fname = os.path.join(d, name)
        if os.path.isfile(fname):
            LOG.i(f"found {fname}")
            return fname
    LOG.f(f"file {name} not found in {dirs}")


def install_mkl(root_folder):
    url = "https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    filename = "mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz", ""))

    if not os.path.isfile(os.path.join(dirname, "examples", "test")):
        LOG.i("Downloading mkl...")
        download_url_to_local(url, filename, root_folder,
                              "47187284ede27ad3bd64b5f0e7d5e730")
        import tarfile

        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)

        assert 0 == os.system(f"cd {dirname}/examples && "
                              f"{cc_path} -std=c++14 cpu_cnn_inference_f32.cpp -Ofast -lmkldnn -I ../include -L ../lib -o test && LD_LIBRARY_PATH=../lib/ ./test")


def setup_mkl():
    global mkl_ops, use_mkl
    use_mkl = os.environ.get("use_mkl", "1") == "1"
    mkl_ops = None
    if not use_mkl:
        return
    mkl_include_path = os.environ.get("mkl_include_path")
    mkl_lib_path = os.environ.get("mkl_lib_path")

    if mkl_lib_path is None or mkl_include_path is None:
        mkl_install_sh = os.path.join(jittor_path, "script", "install_mkl.sh")
        LOG.v("setup mkl...")
        # mkl_path = os.path.join(cache_path, "mkl")
        # mkl_path decouple with cc_path
        from pathlib import Path
        mkl_path = os.path.join(str(Path.home()), ".cache", "jittor", "mkl")

        make_cache_dir(mkl_path)
        install_mkl(mkl_path)
        mkl_home = ""
        for name in os.listdir(mkl_path):
            if name.startswith("mkldnn_lnx") and os.path.isdir(os.path.join(mkl_path, name)):
                mkl_home = os.path.join(mkl_path, name)
                break
        assert mkl_home != ""
        mkl_include_path = os.path.join(mkl_home, "include")
        mkl_lib_path = os.path.join(mkl_home, "lib")

    mkl_lib_name = os.path.join(mkl_lib_path, "libmkldnn.so")
    assert os.path.isdir(mkl_include_path)
    assert os.path.isdir(mkl_lib_path)
    assert os.path.isfile(mkl_lib_name)
    LOG.v(f"mkl_include_path: {mkl_include_path}")
    LOG.v(f"mkl_lib_path: {mkl_lib_path}")
    LOG.v(f"mkl_lib_name: {mkl_lib_name}")
    # We do not link manualy, link in custom ops
    # ctypes.CDLL(mkl_lib_name, dlopen_flags)

    mkl_op_dir = os.path.join(jittor_path, "extern", "mkl", "ops")
    mkl_op_files = [os.path.join(mkl_op_dir, name)
                    for name in os.listdir(mkl_op_dir)]
    mkl_ops = compile_custom_ops(mkl_op_files,
                                 extra_flags=f" -I'{mkl_include_path}' -L'{mkl_lib_path}' -lmkldnn -Wl,-rpath='{mkl_lib_path}' ")
    LOG.vv("Get mkl_ops: "+str(dir(mkl_ops)))


def install_cub(root_folder):
    url = "https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz"
    filename = "cub-1.8.0.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz", ""))

    if not os.path.isfile(os.path.join(dirname, "examples", "test")):
        LOG.i("Downloading cub...")
        download_url_to_local(url, filename, root_folder,
                              "9203ea2499b56782601fddf8a12e9b08")
        import tarfile

        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)
        assert 0 == os.system(f"cd {dirname}/examples && "
                              f"{nvcc_path} device/example_device_radix_sort.cu -O2 -I.. -o test")
        if core.get_device_count():
            assert 0 == os.system(f"cd {dirname}/examples && ./test")
    return dirname


def setup_cub():
    from pathlib import Path
    cub_path = os.path.join(str(Path.home()), ".cache", "jittor", "cub")
    cub_home = install_cub(cub_path)
    setup_cuda_lib("cub", link=False, extra_flags=f"-I{cub_home}")


def setup_cuda_extern():
    if not has_cuda:
        return
    LOG.vv("setup cuda extern...")
    cache_path_cuda = os.path.join(cache_path, "cuda-11.0")
    cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    make_cache_dir(cache_path_cuda)
    cuda_extern_src = os.path.join(jittor_path, "extern", "cuda", "src")
    cuda_extern_files = [os.path.join(cuda_extern_src, name)
                         for name in os.listdir(cuda_extern_src)]
    so_name = os.path.join(cache_path_cuda, "cuda_extern.so")
    compile(cc_path, cc_flags +
            f" -I'{cuda_include}' ", cuda_extern_files, so_name)
    ctypes.CDLL(so_name, dlopen_flags)

    try:
        setup_cub()
    except Exception as e:
        import traceback
        line = traceback.format_exc()
        LOG.w(f"CUDA found but cub is not loaded:\n{line}")

    libs = ["cublas", "cudnn", "curand"]
    for lib_name in libs:
        try:
            setup_cuda_lib(lib_name)
        except Exception as e:
            import traceback
            line = traceback.format_exc()
            LOG.w(f"CUDA found but {lib_name} is not loaded:\n{line}")


def setup_cuda_lib(lib_name, link=True, extra_flags=""):
    globals()[lib_name+"_ops"] = None
    globals()[lib_name] = None
    if not has_cuda:
        return
    LOG.v(f"setup {lib_name}...")

    culib_path = os.path.join(cuda_lib, f"lib{lib_name}.so")
    jt_cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    jt_culib_include = os.path.join(
        jittor_path, "extern", "cuda", lib_name, "inc")

    link_flags = ""
    if link:
        extra_include_path = os.path.abspath(os.path.join(
            cuda_include, "..", "targets/x86_64-linux/include"))
        extra_lib_path = os.path.abspath(os.path.join(
            cuda_lib, "..", "targets/x86_64-linux/lib"))
        cuda_include_name = search_file(
            [cuda_include, extra_include_path, "/usr/include", "/usr/local/cuda-11.0/include"], lib_name+".h")
        culib_path = search_file(
            [cuda_lib, extra_lib_path, "/usr/lib/x86_64-linux-gnu"], f"lib{lib_name}.so")
        # dynamic link cuda library
        ctypes.CDLL(culib_path, dlopen_flags)
        link_flags = f"-l{lib_name} -L'{cuda_lib}'"

    # find all source files
    culib_src_dir = os.path.join(jittor_path, "extern", "cuda", lib_name)
    culib_src_files = []
    for r, _, f in os.walk(culib_src_dir):
        for fname in f:
            culib_src_files.append(os.path.join(r, fname))
    if len(culib_src_files) == 0:
        return

    # compile and get operators
    culib = compile_custom_ops(culib_src_files, return_module=True,
                               extra_flags=f" -I'{jt_cuda_include}' -I'{jt_culib_include}' {link_flags} {extra_flags} ")
    culib_ops = culib.ops
    globals()[lib_name+"_ops"] = culib_ops
    globals()[lib_name] = culib
    LOG.vv(f"Get {lib_name}_ops: "+str(dir(culib_ops)))


def install_cutt(root_folder):
    # Modified from: https://github.com/ap-hynninen/cutt
    url = "https://github.com/Jittor/cutt/archive/master.zip"

    filename = "cutt-master.zip"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".zip", ""))
    true_md5 = "a6f4f7f75310a69b131e21f1ebec768a"

    if os.path.exists(fullname):
        md5 = run_cmd('md5sum '+fullname).split()[0]
        if md5 != true_md5:
            os.remove(fullname)
            shutil.rmtree(dirname)
    if not os.path.isfile(os.path.join(dirname, "bin", "cutt_test")):
        LOG.i("Downloading cutt...")
        download_url_to_local(url, filename, root_folder, true_md5)

        import zipfile

        zf = zipfile.ZipFile(fullname)
        try:
            zf.extractall(path=root_folder)
        except RuntimeError as e:
            print(e)
            raise
        zf.close()

        LOG.i("installing cutt...")
        run_cmd(f"make", cwd=dirname)
    return dirname


def setup_cutt():
    global cutt_ops, use_cutt
    if not has_cuda:
        use_cutt = False
        return
    use_cutt = os.environ.get("use_cutt", "1") == "1"
    cutt_ops = None
    if not use_cutt:
        return
    cutt_include_path = os.environ.get("cutt_include_path")
    cutt_lib_path = os.environ.get("cutt_lib_path")

    if cutt_lib_path is None or cutt_include_path is None:
        LOG.v("setup cutt...")
        # cutt_path decouple with cc_path
        from pathlib import Path
        cutt_path = os.path.join(str(Path.home()), ".cache", "jittor", "cutt")

        make_cache_dir(cutt_path)
        install_cutt(cutt_path)
        cutt_home = os.path.join(cutt_path, "cutt-master")
        cutt_include_path = os.path.join(cutt_home, "src")
        cutt_lib_path = os.path.join(cutt_home, "lib")

    cutt_lib_name = os.path.join(cutt_lib_path, "libcutt.so")
    assert os.path.isdir(cutt_include_path)
    assert os.path.isdir(cutt_lib_path)
    assert os.path.isfile(cutt_lib_name), cutt_lib_name
    LOG.v(f"cutt_include_path: {cutt_include_path}")
    LOG.v(f"cutt_lib_path: {cutt_lib_path}")
    LOG.v(f"cutt_lib_name: {cutt_lib_name}")
    # We do not link manualy, link in custom ops
    ctypes.CDLL(cutt_lib_name, dlopen_flags)

    cutt_op_dir = os.path.join(jittor_path, "extern", "cuda", "cutt", "ops")
    cutt_op_files = [os.path.join(cutt_op_dir, name)
                     for name in os.listdir(cutt_op_dir)]
    cutt_ops = compile_custom_ops(cutt_op_files,
                                  extra_flags=f" -I'{cutt_include_path}'")
    LOG.vv("Get cutt_ops: "+str(dir(cutt_ops)))


def install_nccl(root_folder):
    url = "https://github.com/NVIDIA/nccl/archive/v2.6.4-1.tar.gz"

    filename = "nccl.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, "nccl-2.6.4-1")
    true_md5 = "38d7a9e98d95a99df0a4f1ad6fb50fa7"

    if os.path.exists(fullname):
        md5 = run_cmd('md5sum '+fullname).split()[0]
        if md5 != true_md5:
            os.remove(fullname)
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
    if not os.path.isfile(os.path.join(dirname, "build", "lib", "libnccl.so")):
        LOG.i("Downloading nccl...")
        download_url_to_local(url, filename, root_folder, true_md5)

        if core.get_device_count() == 0:
            return
        if not inside_mpi():
            return

        import tarfile
        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)

        LOG.i("installing nccl...")
        arch_flag = f" -arch={','.join(map(lambda x:'sm_'+str(x),flags.cuda_archs))} "
        run_cmd(
            f"make -j8 src.build CUDA_HOME='{cuda_home}' NVCC_GENCODE='{arch_flag}' ", cwd=dirname)
    return dirname


def setup_nccl():
    global nccl_ops, use_nccl
    use_nccl = os.environ.get("use_nccl", "1") == "1"
    nccl_ops = None
    if not has_cuda or not has_mpi:
        use_nccl = False
        return
    if not use_nccl:
        return
    nccl_include_path = os.environ.get("nccl_include_path")
    nccl_lib_path = os.environ.get("nccl_lib_path")

    if nccl_lib_path is None or nccl_include_path is None:
        LOG.v("setup nccl...")
        # nccl_path decouple with cc_path
        from pathlib import Path
        nccl_path = os.path.join(str(Path.home()), ".cache", "jittor", "nccl")

        make_cache_dir(nccl_path)
        nccl_home = install_nccl(nccl_path)
        if nccl_home is None:
            return
        nccl_include_path = os.path.join(nccl_home, "build", "include")
        nccl_lib_path = os.path.join(nccl_home, "build", "lib")

    if not inside_mpi():
        return

    nccl_lib_name = os.path.join(nccl_lib_path, "libnccl.so")
    assert os.path.isdir(nccl_include_path)
    assert os.path.isdir(nccl_lib_path)
    assert os.path.isfile(nccl_lib_name), nccl_lib_name
    LOG.v(f"nccl_include_path: {nccl_include_path}")
    LOG.v(f"nccl_lib_path: {nccl_lib_path}")
    LOG.v(f"nccl_lib_name: {nccl_lib_name}")
    # We do not link manualy, link in custom ops
    ctypes.CDLL(nccl_lib_name, dlopen_flags)

    nccl_src_dir = os.path.join(jittor_path, "extern", "cuda", "nccl")
    nccl_src_files = []
    for r, _, f in os.walk(nccl_src_dir):
        for fname in f:
            nccl_src_files.append(os.path.join(r, fname))

    nccl_ops = compile_custom_ops(nccl_src_files,
                                  extra_flags=f" -I'{nccl_include_path}' {mpi_compile_flags} ")
    LOG.vv("Get nccl_ops: "+str(dir(nccl_ops)))


def manual_link(flags):
    lib_dirs = []
    libs = []
    for f in flags.split():
        if f.startswith("-l"):
            libs.append(f[2:])
        elif f.startswith("-L"):
            lib_dirs.append(f[2:])
    LOG.v("manual_link:", flags)
    LOG.v("lib_dirs:", lib_dirs)
    LOG.v("libs:", libs)
    for lib in libs:
        for d in lib_dirs:
            libname = os.path.join(d, f"lib{lib}.so")
            if os.path.isfile(libname):
                LOG.v("link:", libname)
                ctypes.CDLL(libname, dlopen_flags)
                break


def inside_mpi():
    return "OMPI_COMM_WORLD_SIZE" in os.environ


def setup_mpi():
    global mpi_ops, mpi, use_mpi
    global mpicc_path, has_mpi
    use_mpi = os.environ.get("use_mpi", "1") == "1"
    mpi_ops = None
    mpi = None
    has_mpi = False
    mpicc_path = env_or_try_find('mpicc_path', 'mpicc')
    if mpicc_path == "":
        LOG.i("mpicc not found, distribution disabled.")
        use_mpi = False
    else:
        use_mpi = True
        has_mpi = True
    if not use_mpi:
        return

    global mpi_compile_flags, mpi_link_flags, mpi_flags
    mpi_compile_flags = run_cmd(mpicc_path+" --showme:compile")
    mpi_link_flags = run_cmd(mpicc_path+" --showme:link")
    mpi_flags = mpi_compile_flags + " " + mpi_link_flags
    LOG.v("mpi_flags: "+mpi_flags)

    # find all source files
    mpi_src_dir = os.path.join(jittor_path, "extern", "mpi")
    mpi_src_files = []
    for r, _, f in os.walk(mpi_src_dir):
        for fname in f:
            mpi_src_files.append(os.path.join(r, fname))

    # mpi compile flags add for nccl
    mpi_compile_flags += f" -I'{os.path.join(mpi_src_dir, 'inc')}' "
    mpi_compile_flags = mpi_compile_flags.replace("-pthread", "")

    mpi_version = get_version(mpicc_path)
    if mpi_version.startswith("(1.") or mpi_version.startswith("(2."):
        # mpi version 1.x need to link like this
        manual_link(mpi_flags)
    # mpi(4.x) cannot use deepbind, it need to
    # share the 'environ' symbol.
    mpi = compile_custom_ops(mpi_src_files,
                             extra_flags=f" {mpi_flags} ", return_module=True,
                             dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW, gen_name_="jittor_mpi_core")
    mpi_ops = mpi.ops
    LOG.vv("Get mpi: "+str(mpi.__dict__.keys()))
    LOG.vv("Get mpi_ops: "+str(mpi_ops.__dict__.keys()))

    def warper(func):
        def inner(self, *args, **kw):
            return func(self, *args, **kw)
        inner.__doc__ = func.__doc__
        return inner
    for k in mpi_ops.__dict__:
        if not k.startswith("mpi_"):
            continue
        if k == "mpi_test":
            continue
        setattr(core.Var, k, warper(mpi_ops.__dict__[k]))


setup_mpi()
in_mpi = inside_mpi()
rank = mpi.world_rank() if in_mpi else 0
setup_nccl()

setup_cutt()
setup_mkl()

setup_cuda_extern()
