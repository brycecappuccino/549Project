#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>
#include <iostream>
#include <future>
#include <fstream>

using namespace std;

class Problem {
    public:
        virtual ~Problem() = default;
        virtual int maxN() const = 0;

        virtual vector<int> generateInput(int n) = 0;
        virtual void runSerial(const vector<int>& data) = 0;
        virtual void runParallel(const vector<int>& data) = 0;
        virtual const char* name() const = 0;
};

class MergeSort : public Problem {
    private:
    const int PARALLEL_THRESHOLD = 10000;

    void serialMergeRanges(vector<int>& v, int left, int mid, int right) const {
        vector<int> A(v.begin() + left, v.begin() + mid);
        vector<int> B(v.begin() + mid, v.begin() + right);
        int total = A.size() + B.size();

        vector<int> output;
        output.resize(total);

        auto mergeTask = [&](int a_start, int a_end, int b_start, int b_end, int out_start) {
            merge(A.begin() + a_start, A.begin() + a_end,
                       B.begin() + b_start, B.begin() + b_end,
                       output.begin() + out_start);
        };

        if (total < PARALLEL_THRESHOLD) {
            merge(A.begin(), A.end(), B.begin(), B.end(), output.begin());
        } else {
            int a_mid = A.size() / 2;
            int b_split = lower_bound(B.begin(), B.end(), A[a_mid]) - B.begin();
            int left_size = a_mid + b_split;

            mergeTask(0, a_mid, 0, b_split, 0);
            mergeTask(a_mid, A.size(), b_split, B.size(), left_size);
        }

        copy(output.begin(), output.end(), v.begin() + left);
    }

    void parallelMergeRanges(vector<int>& v, int left, int mid, int right) const {
        vector<int> A(v.begin() + left, v.begin() + mid);
        vector<int> B(v.begin() + mid, v.begin() + right);
        int total = A.size() + B.size();

        vector<int> output(total);

        auto mergeTask = [&](int a_start, int a_end, int b_start, int b_end, int out_start) {
            merge(A.begin() + a_start, A.begin() + a_end,
                       B.begin() + b_start, B.begin() + b_end,
                       output.begin() + out_start);
        };

        if (total < PARALLEL_THRESHOLD) {
            merge(A.begin(), A.end(), B.begin(), B.end(), output.begin());
        } else {
            int a_mid = A.size() / 2;
            int b_split = lower_bound(B.begin(), B.end(), A[a_mid]) - B.begin();
            int left_size = a_mid + b_split;

            auto fut = async(launch::async, mergeTask, 0, a_mid, 0, b_split, 0);
            mergeTask(a_mid, A.size(), b_split, B.size(), left_size);
            fut.get();
        }

        copy(output.begin(), output.end(), v.begin() + left);
    }

    public:
        int maxN() const override { return 1000000000; }
        vector<int> generateInput(int n) override
        {
            vector<int> values(n);
        
            // random number generator
            random_device rd;
            mt19937 rng(rd());
            uniform_int_distribution<int> distribution(0, n);
            
            // Fill vector
            for (int& value : values) {
                value = distribution(rng);
            }
        
            return values;
        }
    
        void runSerial(const vector<int>& data) override
        {
            vector<int> v = data;

            function<void(int,int)> sortRec = [&](int left, int right)
            {
                if (right - left <= 1) return;
                
                int mid = left + (right - left) / 2;

                sortRec(left, mid);
                sortRec(mid, right);

                serialMergeRanges(v, left, mid, right);
            };
            sortRec(0, v.size());
        }
    
        void runParallel(const vector<int>& data) override
        {
            vector<int> v = data;
        
            function<void(int,int)> sortRec = [&](int left, int right)
            {
                int size = right - left;
                if (size <= 1) return;
        
                int mid = left + size/2;
                if (size < PARALLEL_THRESHOLD)
                {
                    sortRec(left, mid);
                    sortRec(mid, right);
                }
                else
                {
                    auto fut = async(launch::async, sortRec, left, mid); // SPAWN
                    sortRec(mid, right);
                    fut.get(); // SYNC
                }
                parallelMergeRanges(v, left, mid, right);
            };
        
            sortRec(0, v.size());
        }
    
        const char* name() const override { return "Sorting"; }
};

class Demo : public Problem {
    private:
        const int PARALLEL_THRESHOLD = 10000;

        static void helperFunction(){
            cout<<"helperFunction"<<endl;
        };
    public:
        int maxN() const override { return 1000000000; }

        vector<int> generateInput(int n) override{
            cout << "generateInput" << endl;
            //return;
        };

        void runSerial(const vector<int>& data) override{
            cout << "runSerial" << endl;
        };

        void runParallel(const vector<int>& data)override {
            cout << "runParallel" << endl;
        };

    const char* name() const override { return "placeholder name"; }
};

class CacheObliviousMatrixMultiply : public Problem {
private:
    const int PARALLEL_THRESHOLD = 64;

    // Recursive cache-oblivious matrix multiply
    void multiply(const vector<vector<int>>& A, const vector<vector<int>>& B,
                  vector<vector<int>>& C,
                  int ai, int aj,
                  int bi, int bj,
                  int ci, int cj,
                  int size) {
        if (size == 1) {
            C[ci][cj] += A[ai][aj] * B[bi][bj];
            return;
        }

        int half = size / 2;
        multiply(A, B, C, ai, aj, bi, bj, ci, cj, half);
        multiply(A, B, C, ai, aj + half, bi + half, bj, ci, cj, half);
        multiply(A, B, C, ai, aj, bi, bj + half, ci, cj + half, half);
        multiply(A, B, C, ai, aj + half, bi + half, bj + half, ci, cj + half, half);
        multiply(A, B, C, ai + half, aj, bi, bj, ci + half, cj, half);
        multiply(A, B, C, ai + half, aj + half, bi + half, bj, ci + half, cj, half);
        multiply(A, B, C, ai + half, aj, bi, bj + half, ci + half, cj + half, half);
        multiply(A, B, C, ai + half, aj + half, bi + half, bj + half, ci + half, cj + half, half);
    }

    // Parallel version
    void multiplyParallel(const vector<vector<int>>& A, const vector<vector<int>>& B,
                          vector<vector<int>>& C,
                          int ai, int aj,
                          int bi, int bj,
                          int ci, int cj,
                          int size) {
        if (size == 1) {
            C[ci][cj] += A[ai][aj] * B[bi][bj];
            return;
        }

        int half = size / 2;

        if (size <= PARALLEL_THRESHOLD) {
            //8
            multiply(A, B, C, ai, aj, bi, bj, ci, cj, half);
            multiply(A, B, C, ai, aj + half, bi + half, bj, ci, cj, half);
            multiply(A, B, C, ai, aj, bi, bj + half, ci, cj + half, half);
            multiply(A, B, C, ai, aj + half, bi + half, bj + half, ci, cj + half, half);
            multiply(A, B, C, ai + half, aj, bi, bj, ci + half, cj, half);
            multiply(A, B, C, ai + half, aj + half, bi + half, bj, ci + half, cj, half);
            multiply(A, B, C, ai + half, aj, bi, bj + half, ci + half, cj + half, half);
            multiply(A, B, C, ai + half, aj + half, bi + half, bj + half, ci + half, cj + half, half);
        } else {
            auto f1 = async(launch::async, [&]() {
                multiplyParallel(A, B, C, ai, aj, bi, bj, ci, cj, half);
            });
            auto f2 = async(launch::async, [&]() {
                multiplyParallel(A, B, C, ai, aj + half, bi + half, bj, ci, cj, half);
            });
            auto f3 = async(launch::async, [&]() {
                multiplyParallel(A, B, C, ai, aj, bi, bj + half, ci, cj + half, half);
            });
            auto f4 = async(launch::async, [&]() {
                multiplyParallel(A, B, C, ai, aj + half, bi + half, bj + half, ci, cj + half, half);
            });

            multiplyParallel(A, B, C, ai + half, aj, bi, bj, ci + half, cj, half);
            multiplyParallel(A, B, C, ai + half, aj + half, bi + half, bj, ci + half, cj, half);
            multiplyParallel(A, B, C, ai + half, aj, bi, bj + half, ci + half, cj + half, half);
            multiplyParallel(A, B, C, ai + half, aj + half, bi + half, bj + half, ci + half, cj + half, half);

            f1.get();
            f2.get();
            f3.get();
            f4.get();
        }
    }

public:
    int maxN() const override { return 1000000; }
    vector<int> generateInput(int n) override {
        // n = side length of n x n matrix
        int size = 2 * n * n;
        vector<int> flat(size);

        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<int> dist(0, 100);

        for (int& val : flat) val = dist(rng);
        return flat;
    }

    void runSerial(const vector<int>& data) override {
        int n = sqrt(data.size() / 2);
        vector<vector<int>> A(n, vector<int>(n));
        vector<vector<int>> B(n, vector<int>(n));
        vector<vector<int>> C(n, vector<int>(n, 0));

        for (int i = 0; i < n * n; ++i) {
            A[i / n][i % n] = data[i];
            B[i / n][i % n] = data[i + n * n];
        }

        multiply(A, B, C, 0, 0, 0, 0, 0, 0, n);
    }

    void runParallel(const vector<int>& data) override {
        int n = sqrt(data.size() / 2);
        vector<vector<int>> A(n, vector<int>(n));
        vector<vector<int>> B(n, vector<int>(n));
        vector<vector<int>> C(n, vector<int>(n, 0));

        for (int i = 0; i < n * n; ++i) {
            A[i / n][i % n] = data[i];
            B[i / n][i % n] = data[i + n * n];
        }

        multiplyParallel(A, B, C, 0, 0, 0, 0, 0, 0, n);
    }

    const char* name() const override {
        return "Cache-Oblivious Matrix Multiplication";
    }
};



template <typename Func>
double timedRun(Func&& f) {
    // Log start time
    auto start = chrono::high_resolution_clock::now();
    
    f(); // Run function

    // Log end time
    auto end = chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto delta = chrono::duration_cast<chrono::milliseconds>(end - start);

    return delta.count();
}


void testProblem(Problem& prob) {
    string algoName = prob.name();
    int maxN = prob.maxN();
    string filename = algoName + "_" + to_string(maxN) + ".csv";

    ofstream csvFile(filename);
    csvFile << "N,Serial(ms),Parallel(ms),Speedup\n";

    for (int n = 10; n <= maxN; n *= 10) {
        cout << "Testing " << algoName << " with n = " << n << endl;

        auto data = prob.generateInput(n);
        double tSerial = timedRun([&]() { prob.runSerial(data); });
        double tParallel = timedRun([&]() { prob.runParallel(data); });

        double speedup = tSerial / tParallel;

        cout << "Serial: " << tSerial << " ms\n";
        cout << "Parallel: " << tParallel << " ms\n";
        cout << "Speedup: " << speedup << "\n\n";

        csvFile << n << "," << tSerial << "," << tParallel << "," << speedup << "\n";
    }

    csvFile.close();
}

void testLoop(Problem& prob, int maxN) {
    for(int n = 10; n <= maxN; n = n * 10)
    {
        testProblem(prob, n);
    }   
}

int main() {
    vector<unique_ptr<Problem>> problems;
    problems.emplace_back(make_unique<MergeSort>());
    problems.emplace_back(make_unique<CacheObliviousMatrixMultiply>());

    for (auto& p : problems) {
        testProblem(*p);
    }

    return 0;
}