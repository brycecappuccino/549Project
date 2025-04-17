#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>
#include <iostream>
#include <future>


class Problem {
    public:
        virtual ~Problem() = default;
    
        virtual std::vector<int> generateInput(int n) = 0;
        virtual void runSerial(const std::vector<int>& data) = 0;
        virtual void runParallel(const std::vector<int>& data) = 0;
        virtual const char* name() const = 0;
};

class MergeSort : public Problem {
    private:
        const int PARALLEL_THRESHOLD = 10000;
        static void mergeRanges(std::vector<int>& v, int left, int mid, int right)
        {
            std::vector<int> tmp;
            tmp.reserve(right - left);
        
            int i = left, j = mid;
            while (i < mid && j < right) {
                tmp.push_back(v[i] < v[j] ? v[i++] : v[j++]);
            }

            // copy any leftover bits
            tmp.insert(tmp.end(), v.begin() + i, v.begin() + mid);
            tmp.insert(tmp.end(), v.begin() + j, v.begin() + right);
        
            // write back
            std::copy(tmp.begin(), tmp.end(), v.begin() + left);
        }
    public:
        std::vector<int> generateInput(int n) override
        {
            std::vector<int> values(n);
        
            // random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<int> distribution(0, n);
            
            // Fill vector
            for (int& value : values) {
                value = distribution(rng);
            }
        
            return values;
        }
    
        void runSerial(const std::vector<int>& data) override
        {
            std::vector<int> v = data;

            std::function<void(int,int)> sortRec = [&](int left, int right)
            {
                if (right - left <= 1) return;
                
                int mid = left + (right - left) / 2;

                sortRec(left, mid);
                sortRec(mid, right);

                mergeRanges(v, left, mid, right);
            };
            sortRec(0, v.size());
        }
    
        void runParallel(const std::vector<int>& data) override
        {
            std::vector<int> v = data;
        
            std::function<void(int,int)> sortRec = [&](int left, int right)
            {
                int size = right - left;
                if (size <= 1) return;
        
                int mid = left + size/2;
                if (size < PARALLEL_THRESHOLD)
                {
                    sortRec(left, mid);
                    sortRec(mid, right);
                } else
                {
                    auto fut = std::async(std::launch::async, sortRec, left, mid); // SPAWN
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
    auto start = std::chrono::high_resolution_clock::now();
    
    f(); // Run function

    // Log end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return delta.count();
}


void testProblem(Problem& prob, int n) {
    std::cout << "Testing " << prob.name() << " with n = " << n << std::endl;

    auto data = prob.generateInput(n);
    double tSerial   = timedRun([&]() { prob.runSerial(data); });
    double tParallel = timedRun([&]() { prob.runParallel(data); });

    std::cout << "Serial: " << tSerial << " ms" << std::endl;
    std::cout << "Parallel: " << tParallel << " ms" << std::endl;
    std::cout << "Difference: " << tSerial - tParallel << " ms" << std::endl;
    std::cout << "Speedup: " << tSerial / tParallel << std::endl;
}

void testLoop(Problem& prob, int maxN) {
    for(int n = 10; n <= maxN; n = n * 10)
    {
        testProblem(prob, n);
    }   
}

int main() {
    int maxN = 1000000000;

    // Problem vector
    std::vector<std::unique_ptr<Problem>> problems;
    problems.emplace_back(std::make_unique<MergeSort>());

    // Test each problem
    for (auto& p : problems)
    {
        testLoop(*p, maxN);
    }

    return 0;
}