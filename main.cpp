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

        static void mergeRanges(vector<int>& v, int left, int mid, int right)
        {
            vector<int> tmp;
            tmp.reserve(right - left);
        
            int i = left;
            int j = mid;
            while (i < mid && j < right) {
                if (v[i] < v[j]) {
                    tmp.push_back(v[i]);
                    i++;
                } else {
                    tmp.push_back(v[j]);
                    j++;
                }
            }

            // copy any leftover bits
            tmp.insert(tmp.end(), v.begin() + i, v.begin() + mid);
            tmp.insert(tmp.end(), v.begin() + j, v.begin() + right);
        
            // write back
            copy(tmp.begin(), tmp.end(), v.begin() + left);
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
    
        void runSerial(const std::vector<int>& data) override
        {
            vector<int> v = data;

            function<void(int,int)> sortRec = [&](int left, int right)
            {
                if (right - left <= 1) return;
                
                int mid = left + (right - left) / 2;

                sortRec(left, mid);
                sortRec(mid, right);

                mergeRanges(v, left, mid, right);
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
                mergeRanges(v, left, mid, right);
            };
        
            sortRec(0, v.size());
        }
    
        const char* name() const override { return "Sorting"; }
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