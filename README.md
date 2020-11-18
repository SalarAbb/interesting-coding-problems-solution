# interesting-coding-problems-solution
Solution to interesting coding problems from [here](https://github.com/SalarAbb/interesting-coding-problems).

## 1. Continueous subarray with maximum sum, aka Kardane's algorithm ([link](https://practice.geeksforgeeks.org/problems/kadanes-algorithm/0))
My solution:

Time complextiy: O(n), Space complextiy: O(1)
```
import sys
def max_sum(arr):
    if max(arr) <= 0:
        return max(arr)
        
    max_sum = 0
    max_end_at_here = 0
    for i in range(len(arr)):
        max_end_at_here = max_end_at_here + arr[i]
        if max_end_at_here < 0:
            max_end_at_here = 0
        
        max_sum = max(max_sum, max_end_at_here)
    
    return max_sum        
    
    
i = 0 
cases = {}
for line in sys.stdin:
    line = line.strip()
    if i == 0:
        num_test_cases = int(line)
    else:
        if i % 2 == 0:
            line = line.split(' ')
            cases[int(i/2 - 1)] = [int(j) for j in line]
        
    i += 1
#print(cases)
# now cases is the dictionary of all test cases
for i in range(len(cases)):
    arr_this = cases[i]
    print(max_sum(arr_this))
```    
## 2. Minimum window substring ([link](https://leetcode.com/problems/minimum-window-substring/submissions/))

My solution (another interesting solutions is using 2 pointers to expand from right and contract from left):

Time complextiy: O(S*T), Space complextiy: O(T)
```
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # create a dictionary for recent t
        t_list = list(t)
        t_dict = {i:-1 for i in t}        
        
        n = len(s)
        min_sub_length = 2 * n
        min_win = ""
        
        for i, curr_s in enumerate(list(s)):
            if curr_s in t_dict:
                t_dict[curr_s] = i
            vals = list(t_dict.values())
            if -1 not in vals:
                new_win_min = min(vals)
                new_win_max = max(vals)
                new_win_length = new_win_max - new_win_min + 1
                if new_win_length < min_sub_length:
                    min_win = s[new_win_min:new_win_max + 1]
                    min_sub_length = new_win_length
                    
        print(t_dict)
        return min_win
```
## 3. Count the triplets ([link](https://practice.geeksforgeeks.org/problems/count-the-triplets4615/1)):
My solution:

Time complextiy: O(n^2), Space complextiy: O(n)
```
class Solution:
    
    def countTriplet(self, arr, n):
	    # code here
        count = 0
        for num in arr:
            results = self.two_elem_sum(arr,num)
            count += len(results)
        

        return count    

    def two_elem_sum(self,arr,target):
        # a variation of 2 element sum
        d_obs = {}
        results = []
        for i, num in enumerate(arr):
            d_obs[num] = 1
            if target - num in d_obs and num != target - num:
                if (target-num, num, target) not in results and (num, target-num, target) not in results:
                    results.append( (num, target-num, target) )
        
        return results
```

## 4. Longest palindrome in a string ([link](https://practice.geeksforgeeks.org/problems/longest-palindrome-in-a-string/0)):
My solution:

Time complextiy: O(n^2), Space complextiy: O(1)
```
#code

import sys

def longest_palindrome(s):
    n = len(s)
    # check some default test cases
    if n == 1:
        return s
    if n == 2:
        if s[0] == s[1]:
            return s
        else:
            return s[0]
    # main algorithm        
    length_max = 1
    max_pal = s[0]
    for i in range(1, 2*n - 1):
        # set the left and right pointers 
        if i % 2 == 0:
            left_pointer = int(i/2) - 1
            right_pointer = int(i/2) + 1
        else:
            left_pointer = int((i-1)/2)
            right_pointer = int((i-1)/2) + 1
        # check palindrome    
        while(left_pointer >= 0 and right_pointer <= n - 1):
            if s[left_pointer] == s[right_pointer]:
                length_this = right_pointer - left_pointer + 1
                if length_this > length_max:
                    max_pal = s[left_pointer:right_pointer + 1]
                    length_max = length_this
                left_pointer -= 1
                right_pointer += 1    
            else:
                break
    return max_pal
# get test cases
i = 0
test_cases = {}
for line in sys.stdin:
    line= line.strip()
    if i == 0:
        num_test_cases = int(line)
    else:
        test_cases[i - 1] = line
    i += 1    
# print the output   
for i in range( len( test_cases ) ):
    print(longest_palindrome(test_cases[i]))
```
## 5. Flood fill Algorithm ([link](https://practice.geeksforgeeks.org/problems/flood-fill-algorithm/0)):
My solution:
We run bfs from the given point to paint its neighboring points. we keep track of visited nodes in neighbors.

Time complextiy: O(C), Space complextiy: O(C), where C is the color cloud/flood size
```
#code
import collections
import sys

def get_prop_neighbors(tab, neighbors, p_this, c):
    num_rows = len(tab)
    num_cols = len(tab[0])
    x, y = p_this[0], p_this[1]
    
    prop_neighbors = []
    for possible_point in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
        i, j = possible_point[0], possible_point[1]
        if i in range(num_rows) and j in range(num_cols) and tab[i][j] == c and (i, j) not in neighbors:
            prop_neighbors.append((i,j))
    
    return prop_neighbors        
    

def get_neighbors_and_paint(tab, x, y, c_new):
    # get the color
    c = tab[x][y]
    # get all the cloud neighbors by the same color
    neighbors = [(x,y)]
    q = [(x,y)]   
    
    while(q != []):
        num_this = len(q)
        for i in range(num_this):
            # popleft
            p_this = q[0]
            if len(q) == 1:
                q = []
            else:
                q = q[1:]
                
            prop_neighbors_this = get_prop_neighbors(tab, neighbors, p_this, c)
            neighbors += prop_neighbors_this
            q += prop_neighbors_this

    for p in neighbors:
        i, j = p[0], p[1]
        tab[i][j] = c_new
    
    return tab
    
def turn_tab_to_output(tab):
    out = []
    for i in tab:
        out += [str(j) for j in i]
    
    return ' '.join(out)    

# read the input
i = 0
test_cases = {}
for line in sys.stdin:
    line = line.strip()
    if i == 0:
        num_cases = int(line)
    else:
        line = line.split(' ')
        
        if i % 3 == 1:
            case_num = int((i - i%3)/3) 
            test_cases[case_num] = {}
            num_rows, num_cols = int(line[0]), int(line[1])
            test_cases[case_num]['num_rows'] = num_rows
            test_cases[case_num]['num_cols'] = num_cols
        elif i % 3 == 2:
            tab = [[0 for j in range(num_cols)] for m in range(num_rows)]
            for m in range(num_rows):
                for j in range(num_cols):
                    tab[m][j] = int(line[m * num_cols + j])
            test_cases[case_num]['tab'] = tab
         
        elif i % 3 == 0:
            x, y, c_new = int(line[0]), int(line[1]), int(line[2])
            test_cases[case_num]['x'] = x
            test_cases[case_num]['y'] = y
            test_cases[case_num]['c_new'] = c_new
            
    
    i += 1    

# generate the output
for i in range(len(test_cases)):
    tab = get_neighbors_and_paint(test_cases[i]['tab'], test_cases[i]['x'], test_cases[i]['y'], test_cases[i]['c_new'])
    print(turn_tab_to_output(tab))

```
## 6. Find median in a stream ([link](https://practice.geeksforgeeks.org/problems/find-median-in-a-stream/0)):
My solution:
The solution is very simple. We create two heaps: min heap (representing top half of currently observed values) and max heap (representing bottom half of currently observed values). Each time, we observe a new value: 1) we decide it belongs to which heap and insert it there, 2) we always keep len (min heap) equals to len(max heap) or 1 larger; if  after insertion this is not satisfied, we will do it by deletion and insertion. We get the median given the length of the heaps and the first node in min and max heaps.

Time complextiy: O(log(n)), Space complextiy: O(n)
```
#code
import sys
import heapq
import numpy as np
def add_input_to_heaps(heap_min,heap_max,len_heap_min,len_heap_max,inp):
    # we record len_heap_min and len_heap_max to save time
    if len_heap_min == 0:
        heapq.heappush(heap_min, inp)
        len_heap_min += 1
    elif len_heap_max == 0:
        heapq.heappush(heap_max, -inp)
        len_heap_max += 1
    else:    
        min_heap_min = heap_min[0]
        max_heap_max = - heap_max[0]
        
        if inp >= min_heap_min or (inp < min_heap_min and inp > max_heap_max):
            heapq.heappush(heap_min,inp)
            len_heap_min += 1
        else:
            heapq.heappush(heap_max,-inp)
            len_heap_max += 1
        # we always keep len_heap_min equals to len_heap_max or 1 larger
        if len_heap_max == len_heap_min + 1:
            a = heapq.heappop(heap_max)
            max_heap_max = - a
            heapq.heappush(heap_min,max_heap_max)
            len_heap_min += 1
            len_heap_max -= 1
        elif len_heap_min == len_heap_max + 2:     
            min_heap_min = heapq.heappop(heap_min)
            heapq.heappush(heap_max,-min_heap_min)
            len_heap_min -= 1
            len_heap_max += 1
        
    return heap_min, heap_max, len_heap_min, len_heap_max    
        
def get_median_from_heaps(heap_min,heap_max,len_heap_min,len_heap_max):
    if len_heap_min == len_heap_max:
        return int( np.floor(float((heap_min[0] - heap_max[0]))/2) )
    elif len_heap_min == len_heap_max + 1:
        return heap_min[0]
    else:
        assert 1==0, 'There exists a bug'

i = 0    
arr = []
for line in sys.stdin:
    line = line.strip()
    if i == 0:
        num_cases = int(line)
    else:
        arr.append(int(line))
    i += 1

# create output
heap_min = []#heapq.heapify([])
heap_max = []#heapq.heapify([])
len_heap_min = 0
len_heap_max = 0
    
for inp in arr:
    heap_min,heap_max,len_heap_min,len_heap_max = add_input_to_heaps(heap_min,heap_max,len_heap_min,len_heap_max,inp)
    print(get_median_from_heaps(heap_min,heap_max,len_heap_min,len_heap_max))
```
## 7. Relative sorting ([link](https://practice.geeksforgeeks.org/problems/relative-sorting/0))
My solution:
The solution is very simple. We create a hash of new orders in A2. We change A1 elements to their rankings in A2 given the hash. We sort this changed lists, and given another hash from rankings to the actual elements, we get the relatively sorted list. We have to take care of elements that are not in A2 as well, by identifying them and simply sorting them.

Time complextiy: O(nlog(n)), Space complextiy: O(n)
```
import sys

def perform_relative_sorting(A1,A2):
    
    A2_dict = {A2[i]:i for i in range(len(A2))}
    A2_dict_rev = {i:A2[i] for i in range(len(A2))}
    
    A1_rank = []
    A1_left_out = []
    
    for i in A1:
        if i in A2_dict:
            A1_rank.append(A2_dict[i])
        else:
            A1_left_out.append(i)
            
    if A1_left_out == []:
        A1_left_out_sorted =[]
    else:    
        A1_left_out_sorted =sorted(A1_left_out)
    if A1_rank == []:
        A1_rank_sorted = []
    else:    
        A1_rank_sorted = sorted(A1_rank)
        A1_main_sorted = [A2_dict_rev[j] for j in A1_rank_sorted]
    
    return A1_main_sorted + A1_left_out_sorted
    
# get the inputs
i = 0
test_cases = {}
for line in sys.stdin:
    line = line.strip()
    if i == 0:
        num_cases = int(line)
    elif i % 3 == 1:
        case_num = int((i - 1) / 3)
        test_cases[case_num] = {}
    elif i % 3 == 2:         
        line = line.split(' ')
        test_cases[case_num]['A1'] = [int(j) for j in line]
    elif i % 3 == 0:
        line = line.split(' ')
        test_cases[case_num]['A2'] = [int(j) for j in line]
    i += 1
# prepare the output
for i in range(num_cases):
    A1_sorted = perform_relative_sorting(test_cases[i]['A1'],test_cases[i]['A2'])
    A1_sorted_string = ' '.join([str(j) for j in A1_sorted])
    print(A1_sorted_string)
    
```

## 8. Regions cut by slashes ([link](https://leetcode.com/problems/regions-cut-by-slashes/submissions/)):
My solution:
In order to take into account triangles formed by '/' and '\', we divide each square into 4 triangles. Therefore, we have a graph that has 4*n^2 vertices. We then simply run dfs on each vertex and keep track of visited nodes. Within dfs, we get the neighbors of a vertex given the geometrical condiditions imposed by neighboring squares and also '/' and '\'. In dfs, everytime we run out of neighbors, it indicates that we have covered one region. 

Time complextiy: O(n^2), Space complextiy: O(n^2)
```
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        self.grid = grid
        num_rows = len(grid)
        num_cols = num_rows
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        # we create a 3d list (each square is divided to 4 sub-triangles, and we perform dfs to find
        # connected segments, the square triangles starting from top, clockwise are 0, 1, 2, 3)        
        
        self.visited =[ [ [0  for tr in range(4)] for j in range(num_cols)] for i in range(num_rows)]
        num_sections = 0
        for i in range(num_rows):
            for j in range(num_cols):
                for tr in range(4):
                    if self.visited[i][j][tr] == 0:
                        # visit all connected nodes
                        self.dfs(i,j,tr)
                        # add one section
                        num_sections += 1
                        #print('added')
        
        return num_sections
        
    
    def dfs(self,i,j,tr):
        if i not in range(self.num_rows) or j not in range(self.num_cols) or tr not in range(4):
            return
        if self.visited[i][j][tr] == 1:
            return
        #print('{}, {}, {}'.format(i,j,tr))
        # visit this node
        self.visited[i][j][tr] = 1
        
        # go to neighbors
        if tr == 0:
            self.dfs(i-1, j , 2)
        elif tr == 1:    
            self.dfs(i, j+1 , 3)
        elif tr == 2:    
            self.dfs(i+1, j , 0)
        elif tr == 3:    
            self.dfs(i, j-1 , 1)    
        
        if self.grid[i][j] != '/': # '/':
            if tr == 0:
                self.dfs(i, j , 1)
            elif tr == 1:    
                self.dfs(i, j , 0)
            elif tr == 2:    
                self.dfs(i, j , 3)
            elif tr == 3:    
                self.dfs(i, j , 2)
        
        if self.grid[i][j] != '\\': # '\':
            if tr == 0:
                self.dfs(i, j , 3)
            elif tr == 1:    
                self.dfs(i, j , 2)
            elif tr == 2:    
                self.dfs(i, j , 1)
            elif tr == 3:    
                self.dfs(i, j , 0)
        
        return

```
## 9. Longest consecutive sequence ([link](https://leetcode.com/problems/longest-consecutive-sequence/submissions/)):
My solution:
We create a directed acyclic graph (DAG), where a directed link is connected between two consecutive elements in order (e.g., 1->2); this process takes O(n) in time and space. We then run dfs on each node and keep track of 1) visited nodes, 2) the longest consecutive sequence from each node. 

Time complextiy: O(n), Space complextiy: O(n)
```
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if nums == []:
            return 0
        # we assume there is a graph, nodes are connected if there exists a consecutive connection
        
        # create neighbors hash
        self.g = {}
        for i in nums:
            if i + 1 in self.g:
                self.g[i] = i + 1
            else:    
                self.g[i] = None
        
        for i in range(len(nums)-1,-1,-1):
            if nums[i] + 1 in self.g:
                self.g[nums[i]] = nums[i] + 1
            else:
                self.g[nums[i]] = None
        print(self.g)        
        # now find the longest consec seq
        self.visited = {i:0 for i in nums}
        self.longest_path = {i:1 for i in nums}
        
        
        for i in nums:
            self.dfs(i)
        print(self.longest_path)    
        return max(self.longest_path.values()) 

    def dfs(self,i):
        if self.visited[i] == 1:
            return self.longest_path[i]
        else:
            if self.g[i] == None:
                self.visited[i] = 1
                return 1
            else:
                self.visited[i] = 1
                self.longest_path[i] = 1 + self.dfs(self.g[i]) 
                return self.longest_path[i] 
```
## 10. Kth largest element in a stream ([link](https://practice.geeksforgeeks.org/problems/kth-largest-element-in-a-stream/0))
My solution:
We create a min heap containing k largest elements we have observed so far, for every new value we compare it with the head of the heap: if larger we insert it to the heap and if not continue.

Time complextiy: O(n*log(k)), Space complextiy: O(k)
```
#code
import sys
import heapq
def get_kth_largest_element(curr_heap,k,inp):
    n = len(curr_heap)
    if n < k:
        heapq.heappush(curr_heap,inp)
        if n == k - 1:
            return curr_heap, curr_heap[0]
        else:    
            return curr_heap, -1
    else:
        if inp > curr_heap[0]:
            heapq.heappop(curr_heap)
            heapq.heappush(curr_heap,inp)
        
        return curr_heap, curr_heap[0]    

test_cases = {}
i = 0
# prepare the input
for line in sys.stdin:
    line = line.strip()
    
    if i == 0:
        num_cases = int(line)
    else:
        if i % 2 == 1:
            case_num = int( (i-1)/2 )
            test_cases[case_num]={}
            line = line.split(' ')
            test_cases[case_num]['k'] = int(line[0])
            test_cases[case_num]['n'] = int(line[1])
        else:
            line = line.split(' ')
            line = [int(i) for i in line]
            test_cases[case_num]['arr'] = line
    
    
    i += 1    
#print(test_cases)    
# prepare the output

for c in range(num_cases):
    k = test_cases[c]['k']
    arr = test_cases[c]['arr']
    res = []
    curr_heap = []
    for i in arr:
        curr_heap, elem = get_kth_largest_element(curr_heap,k,i)
        res.append(elem)
    
    output_list = [str(i) for i in res]
    print(' '.join(output_list))
 ```

## 11. Special Keyboard ([link](https://practice.geeksforgeeks.org/problems/special-keyboard3018/1))
My solution:
(dynamic programming) We just need to find the optimal last time we copy text into the buffer. To do this, at each N, we loop through all the previous optimal results from 1 to N - 3 (3 -> select, copy and paste), and find the optimum number!

Time complextiy: O(n^2), Space complextiy: O(n)
```
class Solution:
	def optimalKeys(self, N):
	if N in range(1,7):
	    return N

	dp = [0] * (N+1)

	for i in range(1,7):
	    dp[i] = i

	for i in range(7, N + 1):
	    max_a = i
	    for j in range(1,i - 2):
		a_this = (i - j - 1) * dp[j]
		max_a = max(max_a, a_this)

	    dp[i] = max_a

	return dp[N]    
```

## 12. Longest Increasing Subsequence ([link](https://leetcode.com/problems/longest-increasing-subsequence/))**
My solution:
(dynamic programming) We exploit the problem of longest common subsequence! We generate a new sorted list in ascending order (o(nlog(n))) and then find the longest common subsequence in the actual list and the sorted list!

Time complextiy: O(n^2), Space complextiy: O(n^2)
```
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        n = len(nums)
        if n == 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            if nums[1] > nums[0]:
                return 2
            else:
                return 1
            
        num_sorted = list(sorted(set(nums)))
        #print(nums)
        #print(num_sorted)
        return self.find_longest_subseq(nums,num_sorted)
        
    
    def find_longest_subseq(self,s1,s2):
        
        n = len(s1)
        m = len(s2)
        
        dp = [[0 for j in range(m)] for i in range(n)]
        
        # initialization
        i = 0
        flag = 0
        while (i<n and flag == 0):        
            if s1[i] == s2[0]:         
                flag = 1
                for j in range(i,n):
                    dp[j][0] = 1                
            i += 1
            
        i = 0
        flag = 0    
        while (i<m and flag == 0):        
            if s2[i] == s1[0]:         
                flag = 1
                for j in range(i,m):
                    dp[0][j] = 1                
            i += 1
                
        # the for rows and columns
        
        for i in range(1,n):
            for j in range(1,m):
                if s1[i] == s2[j]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    
        #print(dp)
        return dp[-1][-1]      
```
## 13. Egg Dropping Puzzle([link](https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle-1587115620/1))*
My solution:
The problem consists of several sub problems (dynamic programming)! if we have n eggs and k floors, we can test and see what is the minimum number of worst case attempts if we drop the first egg from floor 1 to k!

Time complextiy: O(n^2*k), Space complextiy: O(n*k)
```
def eggDrop(n, k):
    # code here
    dp = [[0 for i in range(k+1)] for j in range(n+1)]
    
    for i in range(1, k + 1):
        dp[1][i] = i
    
    for j in range(2, n + 1):
        for i in range(1, k + 1):
            if j >= i:
                dp[j][i] = dp[j - 1][i]
            else:
                cand = []
                for div in range(1, i + 1):
                    cand.append(max(1 + dp[j][i - div], 1 + dp[j - 1][ div - 1]))
                
                dp[j][i] = min(cand)
    #print(dp)    
    return dp[n][k]
```
