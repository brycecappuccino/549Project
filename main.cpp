#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>
#include <iostream>
#include <future>

using namespace std;

class Problem {
    public:
        virtual ~Problem() = default;
    
        virtual vector<int> generateInput(int n) = 0;
        virtual void runSerial(const vector<int>& data) = 0;
        virtual void runParallel(const vector<int>& data) = 0;
        virtual const char* name() const = 0;
};

class MergeSort : public Problem {
    private:
        const int PARALLEL_THRESHOLD = 10000;

    void serialMergeRanges(vector<int>& v, int left, int mid, int right) {
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

    void parallelMergeRanges(vector<int>& v, int left, int mid, int right) {
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
        vector<int> generateInput(int n){
            cout << "generateInput" << endl;
            //return;
        };

        void runSerial(const vector<int>& data){
            cout << "runSerial" << endl;
        };

        void runParallel(const vector<int>& data){
            cout << "runParallel" << endl;
        };

    const char* name() const override { return "placeholder name"; }
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


void testProblem(Problem& prob, int n) {
    cout << "Testing " << prob.name() << " with n = " << n << endl;

    auto data = prob.generateInput(n);
    double tSerial   = timedRun([&]() { prob.runSerial(data); });
    double tParallel = timedRun([&]() { prob.runParallel(data); });

    cout << "Serial: " << tSerial << " ms" << endl;
    cout << "Parallel: " << tParallel << " ms" << endl;
    cout << "Difference: " << tSerial - tParallel << " ms" << endl;
    cout << "Speedup: " << tSerial / tParallel << endl;
}

void testLoop(Problem& prob, int maxN) {
    for(int n = 10; n <= maxN; n = n * 10)
    {
        testProblem(prob, n);
    }   
}

int main() {
    int maxN = 10000000;

    // Problem vector
    vector<unique_ptr<Problem>> problems;
    problems.emplace_back(make_unique<MergeSort>());

    // Test each problem
    for (auto& p : problems)
    {
        testLoop(*p, maxN);
    }

    return 0;
}