#!/bin/sh
aclocal && autoconf
(cd nux; sh bootstrap.sh)

