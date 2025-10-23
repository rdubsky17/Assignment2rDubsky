#include <mpi.h>
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Tags are listed in this enum for readablity.
enum { REQUEST = 0, WORK = 1, RESULT = 2, STOP = 3 };

// simple but not perfect prime function for workers that skips even numbers.
int is_prime(int x) {
    if (x < 2)
        return 0;
    if ((x & 1) == 0)
        return x == 2;
    for (int d = 3; d * d <= x; d += 2) {
        if (x % d == 0)
            return 0;
    }
    return 1;
}

long long count_primes_range(int a, int b) {
    long long c = 0;
    for (int k = a; k <= b; ++k) {
        c += is_prime(k);
        return c;
    }
}

void r1_worker(void) {
    while (1) {
        int payload[2] = {0, 0};
        MPI_Status st;

        MPI_Recv(payload, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        if (st.MPI_TAG == STOP)
            break;

        int start = payload[0];
        int end = payload[1];
        long long part = count_primes_range(start, end);

        MPI_Send(&part, 1, MPI_LONG_LONG, 0, RESULT, MPI_COMM_WORLD);
    }
}


void r1_master(int N, int CHUNK, int P) {
    printf("N=%d, P=%d, CHUNK=%d\n", N, P, CHUNK);

    int num_segments = 0;
    for (int s = 2; s <= N; s += CHUNK)
        num_segments++;

    double t0 = MPI_Wtime();

    int next_worker = 1;
    for (int s = 2; s <= N; s += CHUNK) {
        int start = s;
        int end = s + CHUNK - 1;
        if (end > N) end = N;

        int payload[2] = { start, end };
        MPI_Send(payload, 2, MPI_INT, next_worker, WORK, MPI_COMM_WORLD);
        
        //printf("Sending [%d - %d] to worker %d\n", start, end, next_worker);

        next_worker++;
        if (next_worker == P) 
            next_worker = 1; // this is the 'round robin' part
    }

    // Collecting the results from the workers
    long long total = 0;
    for (int i = 0; i < num_segments; ++i) {
        long long part = 0;
        MPI_Status st;
        MPI_Recv(&part, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &st);
        total += part;
    }

    // Tell all workers to stop
    for (int w = 1; w < P; ++w)
        MPI_Send(NULL, 0, MPI_INT, w, STOP, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    // Optional small-N correctness check (do it only when small to avoid long sequential time)
    if (N <= 200000) {
        long long sequentialTotal = count_primes_range(2, N);
        if (sequentialTotal != total) {
            printf("Correctness failed. seq=%lld vs parallel=%lld\n", sequentialTotal, total);
        } else {
            printf("Correctness passed.\n");
        }
    }

    // Final summary
    printf("total primes = %lld\n", total);
    printf("elapsed time = %.6fs\n", t1 - t0);
}

static void r2_worker(void) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while (1) {
        int dummy = 1;
        MPI_Send(&dummy, 1, MPI_INT, 0, REQUEST, MPI_COMM_WORLD);

        int payload[2] = {0,0};
        MPI_Status st;
        MPI_Recv(payload, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

        if (st.MPI_TAG == STOP)
            break;

        int start = payload[0];
        int end = payload[1];
        long long part = count_primes_range(start, end);

        MPI_Send(&part, 1, MPI_LONG_LONG, 0, RESULT, MPI_COMM_WORLD);
    }
}

static void r2_master(int N, int CHUNK, int P) {
    printf("N=%d, P=%d, CHUNK=%d\n", N, P, CHUNK);

    double t0 = MPI_Wtime();

    int next = 2;
    long long total = 0;
    int active_workers = P - 1;

    while (active_workers > 0) {
        MPI_Status st;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

        if (st.MPI_TAG == RESULT) {
            long long part = 0;
            MPI_Recv(&part, 1, MPI_LONG_LONG, st.MPI_SOURCE, RESULT,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += part;
        } 
        else if (st.MPI_TAG == REQUEST) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, REQUEST,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (next <= N) {
                int start = next;
                int end = start + CHUNK - 1;
                if (end > N)
                    end = N;
                int payload[2] = {start, end};
                MPI_Send(payload, 2, MPI_INT, st.MPI_SOURCE, WORK, MPI_COMM_WORLD);
                next = end + 1;
            } else {
                MPI_Send(NULL, 0, MPI_INT, st.MPI_SOURCE, STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }
    }

    double t1 = MPI_Wtime();

    if (N <= 200000) {
        long long sequentialTotal = count_primes_range(2, N);
        if (sequentialTotal != total)
            printf("Correctness failed. seq=%lld vs parallel=%lld\n", sequentialTotal, total);
        else
            printf("Correctness passed.\n");
    }

    printf("total primes = %lld\n", total);
    printf("elapsed time = %.7fs\n", t1 - t0);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int P = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Argument Correctness checks
    if (argc < 4) {
        if (rank == 0)
            printf("Incorrect amount of arguments, check the README and try again.\n");
        MPI_Finalize();
        return 0;
    }
    int N = atoi(argv[1]);
    int CHUNK = atoi(argv[2]);
    char *mode = argv[3];

    if (N < 2 || CHUNK < 1 || P < 2) {
        if (rank == 0)
            printf("Need N>=2, CHUNK>=1, and -np >= 2\n");
        MPI_Finalize();
        return 0;
    }

    // Mode selection between r1 and r2
    if (strcmp(mode, "r1") == 0) { // strcmp uses 0 as "Strings match". Rather backwords in my opinion.
        char host[256];
        host[255] = '\0'; // I dont know if hostnames will be null terminated, added just in case.
        gethostname(host, sizeof(host)-1);
        printf("rank %d/%d on %s\n", rank, P, host);

        if (rank == 0) {
            r1_master(N, CHUNK, P);
        } else {
            r1_worker();
        }
    } else if (strcmp(mode, "r2") == 0) {
        char host[256];
        host[255] = '\0';
        gethostname(host, sizeof(host)-1);
        printf("rank %d/%d on %s\n", rank, P, host);

        if (rank == 0)
            r2_master(N, CHUNK, P);
        else
            r2_worker();
    } else {
        if (rank == 0)
            printf("Unknown mode '%s' (use 'r1 or r2')\n", mode);
    }

    MPI_Finalize();
    return 0;
}
