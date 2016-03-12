cat tut_head.html >$2
sed -f maketut.sed $1 >>$2
cat tut_end.html >>$2

