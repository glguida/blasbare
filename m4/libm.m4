AC_DEFUN([AC_LIBM_DIR],
[
	AC_SUBST(LIBMDIR, $1)
	AC_SUBST(OPENLIBMDIR, $2)
	AC_SUBST(COMPILE_LIBM, ["include "'$(BUILDROOT)'"/mkgen/libm-compile.mk"])
	AC_CONFIG_FILES([mkgen/libm-compile.mk:$1/libm-compile.mk.in])
])
