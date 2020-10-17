# interesting-coding-problems-solution
Solution to interesting coding problems from [here](https://github.com/SalarAbb/interesting-coding-problems).
# interesting-coding-problems
Here I list interesting coding problems I have seen over the internet

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
