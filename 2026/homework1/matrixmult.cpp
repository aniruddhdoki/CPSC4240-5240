/*******************************************************
 * matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
 *******************************************************/

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"

using namespace std;

std::vector<std::vector<int>> sequential_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> sequential_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> sequential_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> sequential_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> sequential_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> sequential_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_i_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_j_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> parallel_ij_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B);
std::vector<std::vector<int>> divide_and_conquer_naive(std::vector<std::vector<int>> A,
std::vector<std::vector<int>> B);

int main() {
    int n;
    std::cin >> n;

    // Create matrices A, B, and C (all n x n)
    std::vector<std::vector<int>> A(n, std::vector<int>(n));
    std::vector<std::vector<int>> B(n, std::vector<int>(n));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> D(n, std::vector<int>(n));
    std::vector<std::vector<int>> E(n, std::vector<int>(n));
    std::vector<std::vector<int>> F(n, std::vector<int>(n, 0));

    // Read matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i][j];
        }
    }

    // Read matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> B[i][j];
        }
    }

    // Read matrix D
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> D[i][j];
        }
    }

    // Read matrix E
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> E[i][j];
        }
    }

    auto startC = chrono::high_resolution_clock::now();
    // TODO (OpenMP): perform matrix multiplication A x B and write into C: C = A x B
    // YOUR OpenMP CODE HERE
    C = parallel_i_ikj(A, B);
    auto endC = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedC = endC - startC;

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    auto startF = chrono::high_resolution_clock::now();
    // TODO (ParlayLib): perform matrix multiplication D x E and write into F: F = D x E
    // YOUR ParlayLib CODE HERE
    F = parallel_ij_kij(D, E);
    auto endF = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedF = endF - startF;

    std::cout << "The resulting matrix F = D x E is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << F[i][j] << " ";
        }
        std::cout << "\n";
    }

    cout << "TIME_C:" << elapsedC.count() << endl;
    cout << "TIME_F:" << elapsedF.count() << endl;

    return 0;
}

// --- Sequential (6) ---
std::vector<std::vector<int>> sequential_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int i = 0; i < a; i++)
        for (int j = 0; j < d; j++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> sequential_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int i = 0; i < a; i++)
        for (int k = 0; k < b; k++)
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> sequential_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int j = 0; j < d; j++)
        for (int i = 0; i < a; i++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> sequential_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int j = 0; j < d; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> sequential_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++)
        for (int i = 0; i < a; i++)
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> sequential_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++)
        for (int j = 0; j < d; j++)
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// --- Parallel i (OpenMP) (6) ---
std::vector<std::vector<int>> parallel_i_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    #pragma omp parallel for
    for (int i = 0; i < a; i++)
        for (int j = 0; j < d; j++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> parallel_i_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    #pragma omp parallel for
    for (int i = 0; i < a; i++)
        for (int k = 0; k < b; k++)
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> parallel_i_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int j = 0; j < d; j++) {
        #pragma omp parallel for
        for (int i = 0; i < a; i++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    }
    return C;
}

std::vector<std::vector<int>> parallel_i_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int j = 0; j < d; j++)
        for (int k = 0; k < b; k++) {
            #pragma omp parallel for
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return C;
}

std::vector<std::vector<int>> parallel_i_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++) {
        #pragma omp parallel for
        for (int i = 0; i < a; i++)
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
    }
    return C;
}

std::vector<std::vector<int>> parallel_i_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++)
        for (int j = 0; j < d; j++) {
            #pragma omp parallel for
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return C;
}

// --- Parallel j (OpenMP) (6) ---
std::vector<std::vector<int>> parallel_j_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int i = 0; i < a; i++) {
        #pragma omp parallel for
        for (int j = 0; j < d; j++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    }
    return C;
}

std::vector<std::vector<int>> parallel_j_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int i = 0; i < a; i++)
        for (int k = 0; k < b; k++) {
            #pragma omp parallel for
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return C;
}

std::vector<std::vector<int>> parallel_j_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    #pragma omp parallel for
    for (int j = 0; j < d; j++)
        for (int i = 0; i < a; i++)
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> parallel_j_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    #pragma omp parallel for
    for (int j = 0; j < d; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<int>> parallel_j_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++)
        for (int i = 0; i < a; i++) {
            #pragma omp parallel for
            for (int j = 0; j < d; j++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return C;
}

std::vector<std::vector<int>> parallel_j_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
    std::vector<std::vector<int>> C(a, std::vector<int>(d, 0));
    for (int k = 0; k < b; k++) {
        #pragma omp parallel for
        for (int j = 0; j < d; j++)
            for (int i = 0; i < a; i++)
                C[i][j] += A[i][k] * B[k][j];
    }
    return C;
}

// --- Parallel i and j (ParlayLib) (6) ---
std::vector<std::vector<int>> parallel_ij_ijk(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: i (outer), j (middle), k (inner)
    parlay::parallel_for(size_t(0), a, [&](size_t i) {
        parlay::parallel_for(size_t(0), d, [&](size_t j) {
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
        }, 0L, false);
    }, 0L, false);
    return C;
}

std::vector<std::vector<int>> parallel_ij_ikj(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: i (outer), k (middle), j (inner)
    parlay::parallel_for(size_t(0), a, [&](size_t i) {
        for (int k = 0; k < b; k++)
            parlay::parallel_for(size_t(0), d, [&](size_t j) {
                C[i][j] += A[i][k] * B[k][j];
            }, 0L, false);
    }, 0L, false);
    return C;
}

std::vector<std::vector<int>> parallel_ij_jik(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: j (outer), i (middle), k (inner)
    parlay::parallel_for(size_t(0), d, [&](size_t j) {
        parlay::parallel_for(size_t(0), a, [&](size_t i) {
            for (int k = 0; k < b; k++)
                C[i][j] += A[i][k] * B[k][j];
        }, 0L, false);
    }, 0L, false);
    return C;
}

std::vector<std::vector<int>> parallel_ij_jki(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: j (outer), k (middle), i (inner)
    parlay::parallel_for(size_t(0), d, [&](size_t j) {
        for (int k = 0; k < b; k++)
            parlay::parallel_for(size_t(0), a, [&](size_t i) {
                C[i][j] += A[i][k] * B[k][j];
            }, 0L, false);
    }, 0L, false);
    return C;
}

std::vector<std::vector<int>> parallel_ij_kij(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: k (outer), i (middle), j (inner)
    for (int k = 0; k < b; k++)
        parlay::parallel_for(size_t(0), a, [&](size_t i) {
            parlay::parallel_for(size_t(0), d, [&](size_t j) {
                C[i][j] += A[i][k] * B[k][j];
            }, 0L, false);
        }, 0L, false);
    return C;
}

std::vector<std::vector<int>> parallel_ij_kji(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B) {
    size_t a = size(A), d = size(B[0]);
    int b = static_cast<int>(size(A[0]));
    std::vector<std::vector<int>> C(static_cast<int>(a), std::vector<int>(static_cast<int>(d), 0));
    // Order: k (outer), j (middle), i (inner)
    for (int k = 0; k < b; k++)
        parlay::parallel_for(size_t(0), d, [&](size_t j) {
            parlay::parallel_for(size_t(0), a, [&](size_t i) {
                C[i][j] += A[i][k] * B[k][j];
            }, 0L, false);
        }, 0L, false);
    return C;
}

std::vector<std::vector<int>> divide_and_conquer_naive(std::vector<std::vector<int>> A,
std::vector<std::vector<int>> B) {
    int a = size(A), b = size(A[0]), d = size(B[0]);
   
    if (a == 2) {
        std::vector<std::vector<int>> C = std::vector<
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ];
    }
}