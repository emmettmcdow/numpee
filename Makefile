CC     = clang
CFLAGS = -Wall -Wextra -Werror -O0 -g

numpee: vec.c
	$(CC) $(CFLAGS) -o numpee.a vec.c

test: vec.c
	$(CC) $(CFLAGS) -DTEST -o test_vec vec.c && ./test_vec

clean:
	rm -f numpee.a test_vec
