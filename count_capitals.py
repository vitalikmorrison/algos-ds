""" Collection of Algorithmic and Data Structures problems, as well as random code snippets """
import collections

count = 0
with open('test_file.txt') as fh:
    count = sum(a_char.isupper() for a_line in fh for a_char in a_line)

print(f"Number of caps is {count}")


"""
Closures
The reason for this is that Python’s closures are late binding.
This means that the values of variables used in closures
are looked up at the time the inner function is called.
So as a result, when any of the functions returned by multipliers() are called,
the value of i is looked up in the surrounding scope at that time.
By then, regardless of which of the returned functions is called,
the for loop has completed and i is left with its final value of 3.
Therefore, every returned function multiplies the value it is passed by 3,
so since a value of 2 is passed in the above code, they all return a value of 6 (i.e., 3 x 2).
"""


def multipliers():
    return [lambda x: i * x for i in range(4)]

print()
print(f'Closure:{[m(2) for m in multipliers()]}')


class DefaultDict(dict):
  def __missing__(self, key):
    return []

d = DefaultDict()
d['florp'] = 127
print(d['test'])

print()
print("LEETCODE PROBLEMS")
print()

"""
Write a function that prints the least integer that is not present in a given list and cannot be represented
by the summation of the sub-elements of the list.

E.g. For a = [1,2,5,7] the least integer not represented by the list or a slice of the list is 4
and if a = [1,2,2,5,7] then the least non-representable integer is 18.
"""


from itertools import combinations
def least_integer_not_sum(input_list):
    # sort the list
    # keep track of sums (number not in sums)

    sum_elements = set()

    for i in range(len(input_list) + 1):
        for a_sum_combination in combinations(input_list, i):
            sum_elements.add(sum(a_sum_combination))

    candidate = input_list[0]

    while True:
        candidate += 1
        if candidate not in sum_elements and candidate not in input_list:
            break

    return candidate


input1 = [1, 2, 5, 7]
input2 = [1, 2, 2, 5, 7]
print(f'1: Last sum for {input1}: {least_integer_not_sum(input1)}')
print(f'1: Last sum for {input2}: {least_integer_not_sum(input2)}')


""" Two Sum """
class Solution:
    """ Time complexity: O(n squared)
        Space complexity: O(1)
    """
    def two_sum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        for ind1 in range(len(nums)):
            for ind2 in range(ind1+1, len(nums)):
                if nums[ind1] == target - nums[ind2]:
                    return [ind1, ind2]

        return []

print()
print('2: Two Sum: ', Solution().two_sum([2, 7, 11, 15], 9))


class Solution:
    """ Time complexity: O(n)
        Space complexity: O(n)
    """
    def two_sum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        values_map = {}

        for ind in range(len(nums)):
            if nums[ind] in values_map:
                continue
            values_map[nums[ind]] = ind

        for ind in range(len(nums)):
            target_val = target - nums[ind]
            if target_val in values_map:
                return [ind, values_map[target_val]]

        return []

print('2: Two Sum', Solution().two_sum([2,7,11,15], 26))


class Solution:
    """ Time complexity: O(n)
        Space complexity: O(n)
    """
    def two_sum3(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        values_map = {}

        for ind, val in enumerate(nums):
            if val in values_map:
                continue
            values_map[val] = ind

            target_val = target - val
            if target_val in values_map:
                return [values_map[target_val], ind]

        return []

print('3: Two Sum', Solution().two_sum3([2,7,11,15], 9))


### Best time to buy and sell stock
# Sell and buy only once
input = [7, 1, 5, 3, 6, 4]


class Solution:
    """ Time Complexity O(n)
        Space Complexity O(1)
    """

    def find_max_profit(self, prices: 'List[int]') -> 'int':
        max_profit = 0

        last_price = input[0]
        lowest_price = last_price

        for price_today in input[1:]:
            if price_today < lowest_price:
                lowest_price = price_today

            else:
                max_profit = max(price_today - lowest_price, max_profit)

        return max_profit

print()
print('3: Max Profit, Buy and Sell Stock ', Solution().find_max_profit(input))


"""
Contains Duplicates
Given an array of integers, find if the array contains any duplicates.

Your function should return true if any value appears at least twice in the array
and it should return false if every element is distinct.
"""


class Solution:
    def contains_duplicate(self, nums: 'List[int]') -> 'bool':
        # num_set = set(nums)
        # if len(num_set) == len(nums):
        #             return False
        # return True

        nums_set = set()

        for i in nums:
            if i in nums_set:
                return True
            nums_set.add(i)

        return False


input1 = [1, 2, 3, 1]
input2 = [1, 2, 3, 4]
print('4: Contains Duplicates: ', Solution().contains_duplicate(input1))
print('4.2: Contains Duplicates: ', Solution().contains_duplicate(input2))


class Solution:
    def contains_duplicates(self, nums: 'List[int]') -> 'bool':
        nums_sorted = sorted(nums)

        for i, num in enumerate(nums_sorted[1:]):
            if num == nums[i-1]:
                return True

        return False

input1 = [1, 2, 3, 1]
print('4: Contains Duplicates after Sorting', Solution().contains_duplicates(input1))


""" Product Array """
class Solution:
    def product_except_self(self, nums: 'List[int]') -> 'List[int]':
        output = [1]*len(nums)

        for i, val in enumerate(nums):
            for output_i, output_val in enumerate(output):
                if output_i != i:
                    output[output_i] *= val

        return output


input1 = [1, 2, 3, 4]
print('5 :Product Except Self: ', Solution().product_except_self(input1))

"""Fun approach
We can utilize properties of log, if input array is [a, b, c, d] then
precalculated_val = log(a* b* c* d) = log(a) + log(b) + log(c) + log(d)
So for output array it would be { antilog [precalculated_val - log(arr[i])] }
"""


"""
6 Minimum on the rotated array - Binary Search Approach
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.
"""


class Solution:
    @staticmethod
    def find_min(nums: 'List[int]') -> 'int':
        mid_point = len(nums)//2

        if nums[mid_point] > nums[mid_point+1]:
            return nums[mid_point+1]
        elif nums[mid_point-1] > nums[mid_point]:
            return nums[mid_point]

        elif nums[0] < nums[mid_point]:
            return Solution.find_min(nums[mid_point:])
        else:
            return Solution.find_min(nums[:mid_point])

input1 = [4, 5, 6, 7, 0, 1, 2]
print('6: Min on Rotated Array : ', Solution.find_min(input1))


""" Container With Most Water
Solution with two pointers """


class Solution:
    def max_area(self, height: 'List[int]') -> 'int':
        maxarea = 0
        l = 0
        r = len(input) - 1

        while l < r:
            maxarea = max(maxarea, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return maxarea

input = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print('7: Container With Most Water: Max Area: ', Solution().max_area(input))


def max_sub_array_sum(array):
    max_so_far = -100000
    max_ending_here = 0

    for i in range(len(array)):

        max_ending_here = max_ending_here + array[i]

        if max_so_far < max_ending_here:
            max_so_far = max_ending_here

        max_ending_here = max(max_ending_here, 0)

    return max_so_far

a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

print("8: Maximum contiguous sum in array is", max_sub_array_sum(a))

print('9: 3sum - 3 elements of array sum up to a value')
"""
Number 9:

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0?
Find all unique triplets in the array which gives the sum of zero.
Step 1: sort the array
Step 2: keep three indexes: i=0, j=i+1, k=len(array) - 1

iterate array over i:

    while j<k:
        if arr[i] + arr[j] + arr[k] == sum:
            add_triplet()
            j++
            k--

        else if arr[i] + arr[j] + arr[k] < sum:
            j++
        else if arr[i] + arr[j] + arr[k] > sum:
            k--

"""


# STRINGS
""" 10: Longest Substring of non-repeating characters
"""


class Solution:
    def length_longest_substring(self, input_string):

        sub = ''
        res = ''
        max_len = 0

        for a_char in input_string:
            if a_char not in sub:
                sub += a_char
                res = sub
                max_len = max(max_len, len(res))
            else:
                cut = sub.index(a_char)
                sub = sub[cut+1:] + a_char
        return max_len, res

print(f'10: Longest substring: {Solution().length_longest_substring("abcabcbb")}')


""" 11 Power function implementation
    Binary Search Approach
"""


class Solution:
    def power_func(self, x, y):
        if y == 0:
            return 1

        else:
            temp = self.power_func(x, int(y/2))
            if y % 2 == 0:
                return temp*temp
            else:
                return temp*temp*x

print(f'11: Power of 2 to 10 is {Solution().power_func(2, 10)}')

"""12 Sqrt Implementation
    Binary Search Approach

"""


class Solution:
    def my_sqrt(self, x: 'int') -> 'int':

        if x in (0, 1):
            return x

        rbound = x
        lbound = 1

        answer = 0

        while lbound <= rbound:
            mid = lbound + (rbound - lbound) // 2

            if mid * mid == x:
                return mid

            elif mid * mid > x:
                rbound = mid - 1

            else:
                lbound = mid + 1
                answer = mid

        return answer

print(f'12: Sqrt of 20 is {Solution().my_sqrt(20)}')


class Solution:
    def set_zeroes(self, matrix: 'List[List[int]]') -> 'None':
        """
            Do not return anything, modify matrix in-place instead.
        """
        no_rows = len(matrix)
        no_cols = len(matrix[0])
        rows, cols = set(), set()

        # Essentially, we mark the rows and columns that are to be made zero
        for i in range(no_rows):
            for j in range(no_cols):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        # Iterate over the array once again and using the rows and cols sets, update the elements
        for i in range(no_rows):
            for j in range(no_cols):
                if i in rows or j in cols:
                    matrix[i][j] = 0

        # Space and time efficiency O(M*N)
        # Better approach - iterate once, set first element of row or column to 0, then iterate again, and set the rest.
        # Space = O(1), time efficiency = M * N

input_mat = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]
Solution().set_zeroes(input_mat)
print(f'13: Zeroed matrix is {input_mat}')


""" Find target value in rotated sorted array"""
" 1: Find the pivot "
" 2: Apply binary search to either section of the array"


class Solution:
    def find_target(self, array: 'List[int]', target: 'int') -> 'int':

        def find_pivot(array, low, high):
            if array[low] < array[high]:
                return -1

            if low == high:
                return low

            mid = int((low+high)/2)

            if mid < high and array[mid] > array[mid+1]:
                return mid
            elif mid > low and array[mid] < array[mid-1]:
                return mid-1
            elif array[low] < array[mid]:
                return find_pivot(array, mid+1, high)
            else:
                return find_pivot(array, low, mid-1)

        def binary_search(array, low, high, key):
            if high < low:
                return -1

            mid = low + (high - low) // 2

            if key == array[mid]:
                return mid
            elif key > array[mid]:
                return binary_search(array, mid+1, high, key)
            else:
                return binary_search(array, low, mid-1, key)

        pivot = find_pivot(array, 0, len(array) - 1)

        if target == array[pivot]:
            return pivot

        elif target < array[0]:
            return binary_search(array, pivot+1, len(array)-1, target)
        else:
            return binary_search(array, 0, pivot-1, target)

input_arr = [3, 4, 5, 6, 7, 0, 1, 2]

print()
print(f'15: Target value 0 in rotated sorted array {Solution().find_target(input_arr, 0)}')
print(f'15: Target value 7 in rotated sorted array {Solution().find_target(input_arr, 7)}')
print(f'15: Target value 1 in rotated sorted array {Solution().find_target(input_arr, 1)}')
print(f'15: Target value 5 in rotated sorted array {Solution().find_target(input_arr, 5)}')
print(f'15: Target value 3 in rotated sorted array {Solution().find_target(input_arr, 3)}')


""" 16 Peak Element """
class Solution:
    def peak_element(self, nums: 'List[int]') -> 'int':

        def find_peak(nums, l, r):
            if l == r:
                return l

            mid = (l + r) // 2

            if nums[mid] > nums[mid + 1]:
                return find_peak(nums, l, mid)
            else:
                return find_peak(nums, mid + 1, r)

        return find_peak(nums, 0, len(nums) - 1)

input_arr = [1, 2, 1, 3, 5, 6, 4]
print()
print(f'16 Peak element in sequence is {Solution().peak_element(input_arr)}')


""" 17 Two Sum Binary """
class Solution:
    def two_sum(self, nums: 'List[int]', target: 'int') -> 'List[int]':

        # find breaking point, where nums[i] > target
        def break_point(nums, l, r):
            if l == r:
                return l

            mid = (l + r) // 2

            if nums[mid] >= target:
                return break_point(nums, 0, mid - 1)
            else:
                return break_point(nums, mid + 1, r)

        bpoint = break_point(nums, 0, len(nums) - 1)

        vals_dict = dict()

        for i in range(bpoint + 1):
            vals_dict[nums[i]] = i
            target_diff = target - nums[i]

            if target_diff in vals_dict:
                return [min(i, vals_dict[target_diff]), max(i, vals_dict[target_diff])]


input_arr = [2, 5, 7, 9, 11, 13, 15, 29]
print()
print(f'17 Two Sum Binary/Hash Solution for 12 on input: {input} : {Solution().two_sum(input_arr, 12)}')


# 18  3Sum Closest
class Solution:
    def three_sum_closest(self, nums: 'List[int]', target: 'int') -> 'int':

        nums_sorted = sorted(nums)
        closest_sum = nums_sorted[0] + nums_sorted[1] + nums_sorted[len(nums)-1]

        for i in range(len(nums)):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                sum_of_three = nums_sorted[i] + nums_sorted[k] + nums_sorted[j]
                if abs(sum_of_three - target) < abs(closest_sum - target):
                    closest_sum = sum_of_three

                if sum_of_three < target:
                    j += 1
                else:
                    k -= 1

        return closest_sum

input_arr = [-1, 2, 1, -4]
print()
print(f'18 3Sum closest to 2 in input : {input} : {Solution().three_sum_closest(input_arr, 2)}')


"""
19  Trapping Rain Water
Algorithm

Initialize left pointer to 0 and right pointer to size-1
While left<right, do:
If height[left] is smaller than height[right]
    If {height[left]}>={left_max}, update {left_max}
    Else add {left_max}-{height[left]} to ans
        Add 1 to left.
Else
    If {height[right]}>={right_max}, update {right_max}
    Else add {right_max}-{height[right]} to ans
    Subtract 1 from right.
"""


class Solution:
    @staticmethod
    def trap_water(nums: 'List[int]') -> 'int':

        sum_trapped = 0
        l = 0
        r = len(nums) - 1

        max_right = 0
        max_left = 0

        while l < r:
            if nums[r] > nums[l]:
                if nums[l] >= max_left:
                    max_left = nums[l]
                else:
                    sum_trapped += max_left - nums[l]
                l += 1

            else:
                if nums[r] >= max_right:
                    max_right = nums[r]
                else:
                    sum_trapped += max_right - nums[r]
                r -= 1

        return sum_trapped


input_arr = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
print()
print(f'19 Water trapped in {input} : {Solution.trap_water(input_arr)}')


class Solution:

    @staticmethod
    def is_palindrome(input_val) -> 'bool':
        return str(input_val) == str(input_val)[::-1]

print(f'20 Palindrome of {21133112}: {Solution.is_palindrome(21133112)}')


print()
print('### LINKED LIST ###')
class ListNode:
    def __init__(self, data=None):
        self.val = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def print_list(self, head=None):
        current = head if head else self.head

        output = []

        while current:
            output.append(current.val)
            current = current.next

        return ', '.join([str(i) for i in output])

    def insert_item(self, item):
        item.next = self.head
        self.head = item

        return self.head

    def reverse_list(self):
        current = self.head
        previous = None

        while current:
            temp = current.next
            current.next = previous
            previous = current
            current = temp

        self.head = previous

    def reverse_list_recursively(self, head):
        if not head or not head.next:
            return head

        p = self.reverse_list_recursively(head.next)
        head.next.next = head
        head.next = None

        return p


llist = LinkedList()
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))


print(f'21 Reverse Linked List: Original: {llist.print_list()}')
llist.reverse_list()
print(f'21 Reverse Linked List: Reversed: {llist.print_list()}')
llist.head = llist.reverse_list_recursively(llist.head)
print(f'21 Reverse Linked List: Recursive Back Original: {llist.print_list()}')

"""You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse
order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
"""


ll1 = LinkedList()
ll1.insert_item(ListNode(3))
ll1.insert_item(ListNode(4))
ll1.insert_item(ListNode(2))

ll2 = LinkedList()
ll2.insert_item(ListNode(4))
ll2.insert_item(ListNode(6))
ll2.insert_item(ListNode(5))


class Solution:
    def add_two_numbers(self, l1: 'ListNode', l2: 'ListNode') -> 'ListNode':
        res_head = ListNode(0)
        p = l1
        q = l2
        current = res_head
        carry = 0

        while p or q:
            x = p.val if p is not None else 0
            y = q.val if q is not None else 0

            sum_int = x + y + carry
            carry = sum_int // 10
            current.next = ListNode(sum_int % 10)
            current = current.next

            if p is not None:
                p = p.next

            if q is not None:
                q = q.next

        if carry > 0:
            current.next = ListNode(carry)

        return res_head.next


def print_ll(head):
    if not head:
        return ''
    output = f'{head.val}'

    current = head.next
    while current is not None:
        output += f', {current.val}'
        current = current.next

    return output

print(f'22 Add Two Numbers:"{print_ll(ll1.head)}" : "{print_ll(ll2.head)}": '
      f'"{print_ll(Solution().add_two_numbers(ll1.head, ll2.head))}"')


# Reverse between m and n
ll1 = LinkedList()
ll1.insert_item(ListNode(5))
ll1.insert_item(ListNode(4))
ll1.insert_item(ListNode(3))
ll1.insert_item(ListNode(2))
ll1.insert_item(ListNode(1))


def reverse_list(head, m, n):
    current = head
    previous = None

    left_pointer = 0
    reversed_items = n - m + 1

    while current is not None:
        while left_pointer < m - 1:
            previous = current
            current = current.next
            left_pointer += 1

        # from where to join to the next element
        left_current = previous

        # from where to join to the rest of the list
        right_current = current

        for i in range(reversed_items):
            temp = current.next
            current.next = previous
            previous = current
            current = temp

        left_current.next = previous
        right_current.next = current

        return head

print(f'23 Reverse Linked List '
      f'between 2 and 4 elements: Reversed: {print_ll(reverse_list(ll1.head, 2, 4))}')

ll1 = LinkedList()
ll1.insert_item(ListNode(1))
ll1.insert_item(ListNode(2))
ll1.insert_item(ListNode(2))
ll1.insert_item(ListNode(1))


""" Palindrome
    Naive Solution
"""
class Solution:
    def is_palindrome(self, llist):
        current = llist.head
        lstr = ''

        while current is not None:
            lstr += str(current.val)
            current = current.next

        reverse_lstr = lstr[::-1]
        if lstr == reverse_lstr:
            return True
        return False


# Reverse linked list, then traverse both
class Solution:
    def is_palindrome(self, head: 'ListNode') -> 'bool':

        def _reverse_list(head):
            if not head or not head.next:
                return head

            new_head = _reverse_list(head.next)
            head.next.next = head
            head.next = None

            return new_head

        def _copy_llist(head):
            new_head = ListNode(head.val)
            current = new_head
            current_orig = head

            while current_orig.next:
                current.next = ListNode(current_orig.next.val)
                current = current.next
                current_orig = current_orig.next

            return new_head

        if not head or not head.next:
            return True

        second_list = _copy_llist(head)
        rev_head = _reverse_list(second_list)

        while rev_head and head:
            if rev_head.val != head.val:
                return False
            rev_head, head = rev_head.next, head.next

        return True


print(f'24 Palindrome linked list'
      f' {Solution().is_palindrome(ll1.head)}')


""" Reorder linked list O...N elements to the following format: 0, N, 1, N-1, 2, N-2 ...'
    Solution:
1. Break in half
2. Reverse second half
3. Join lists

"""
class Solution:
    """
        brilliant solution
    """
    def reorder_list(self, head):
        # let's get the pointer to the middle of the list
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reorder second half of the list
        current = slow
        prev = None
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp

        first, second = head, prev

        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next


ll1 = LinkedList()
ll1.insert_item(ListNode(5))
ll1.insert_item(ListNode(4))
ll1.insert_item(ListNode(3))
ll1.insert_item(ListNode(2))
ll1.insert_item(ListNode(1))

Solution().reorder_list(ll1.head)
print(f'25 Reorder linked list O...N elements to the following format: 0, N, 1, N-1, 2, N-2 ...'
      f' {print_ll(ll1.head)}')


""" Sort red, white, blue colors """
class Solution:
    def sort_colors_simple(self, nums):
        red = 0
        white = 1
        blue = 2

        r = 0
        w = 0
        b = 0

        for i in nums:
            if i == red:
                r += 1

            elif i == white:
                w += 1

            elif i == blue:
                b += 1

        for k in range(r):
            nums[k] = red
        for k in range(w):
            nums[k + r] = white
        for k in range(b):
            nums[k + r + w] = blue

        return nums

    def sort_colors2(self, nums):
        red, white, blue = 0, 0, len(nums)-1

        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
        return nums


input = [2, 0, 2, 1, 1, 0]
print(f'26 Sort Colors - mulptiple pointers'
      f' {Solution().sort_colors2(input)}')


"""Two Sum - Sorted Array"""
class Solution:
    def two_sum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        left = 0
        right = len(nums) - 1

        while nums[right] >= target:
            right -= 1

        while left <= right:
            target_sum = nums[right] + nums[left]
            if target_sum == target:
                return [left, right]
            elif target_sum > target:
                right -= 1

            else:
                left += 1

        return [-1, -1]


input_arr = [2, 7, 11, 15]
print(f'27 Two Sum II - Sorted Array mulptiple pointers'
      f' {Solution().two_sum(input_arr, 18)}')

"""
Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]


Output: [2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Note:

Each element in the result must be unique.
The result can be in any order.

"""


class Solution:
    def array_intersection(self, nums1: 'List[int]', nums2: 'List[int]') -> 'List[int]':
        s_nums1 = sorted(nums1)
        s_nums2 = sorted(nums2)

        nums1_pointer = 0
        nums2_pointer = 0

        intersection = set()

        while nums1_pointer < len(nums1) and nums2_pointer < len(nums2):
            if s_nums1[nums1_pointer] == s_nums2[nums2_pointer]:
                intersection.add(s_nums1[nums1_pointer])
                nums1_pointer += 1
                nums2_pointer += 1

            elif s_nums1[nums1_pointer] <= s_nums2[nums2_pointer]:
                nums1_pointer += 1

            else:
                nums2_pointer += 1

        return list(intersection)


nums1 = [1, 2, 2, 1]; nums2 = [2, 2]
nums3 = [4, 9, 5]; nums4 = [9, 4, 9, 8, 4]

print(f'28 Two Arrays Intersection - Sorted Array mulptiple pointers'
      f' {Solution().array_intersection(nums3, nums4)}')

"""
Write a function that takes a string as input and reverses only the vowels of a string.

Example 1:

Input: "hello"
Output: "holle"
Example 2:

Input: "leetcode"
Output: "leotcede"
Note:
The vowels does not include the letter "y"
"""


class Solution:
    def reverse_vowels(self, s: 'str') -> 'str':
        left = 0
        right = len(s) - 1

        lresult = ''
        rresult = ''

        while left <= right:
            if s[left] in 'aeiou':
                if s[right] in 'aeiou':
                    lresult += s[right]
                    rresult = s[left] + rresult
                    right -= 1
                    left += 1
                else:
                    rresult = s[right] + rresult
                    right -= 1

            else:
                lresult += s[left]
                left += 1

        return lresult + rresult

reverse_input = 'leetcode'
print(f'29 Reverse vowels in {reverse_input}:'
      f' {Solution().reverse_vowels(reverse_input)}')

"""
Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]

"""

nums1 = [1, 2, 2, 1]; nums2 = [2, 2]
nums3 = [4, 9, 5]; nums4 = [9, 4, 9, 8, 4]


class Solution:
    def intersect(self, nums1: 'List[int]', nums2: 'List[int]') -> 'List[int]':
        snums1 = sorted(nums1)
        snums2 = sorted(nums2)
        intersection = list()

        nums1_pointer = 0
        nums2_pointer = 0

        while nums1_pointer < len(nums1) and nums2_pointer < len(nums2):

            if snums1[nums1_pointer] == snums2[nums2_pointer]:
                intersection.append(snums1[nums1_pointer])
                nums1_pointer += 1
                nums2_pointer += 1

            elif snums1[nums1_pointer] > snums2[nums2_pointer]:
                nums2_pointer += 1

            else:
                nums1_pointer += 1

        return intersection

print(f'30 Intersection in  2 arrays - repeated elements{nums3, nums4}:'
      f' {Solution().intersect(nums3, nums4)}')
print(f'30 Intersection in  2 arrays - repeated elements{nums1, nums2}:'
      f' {Solution().intersect(nums1, nums2)}')


"""125
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Note: For the purpose of this problem, we define empty string as valid palindrome.

Example 1:

Input: "A man, a plan, a canal: Panama"
Output: true
Example 2:

Input: "race a car"
Output: false
"""


class Solution:
    def is_palindrome(self, s: 'str') -> 'bool':
        l_pointer = 0
        r_pointer = len(s) - 1

        while l_pointer <= r_pointer:
            if not s[l_pointer].isalpha():
                l_pointer += 1
                continue

            if not s[r_pointer].isalpha():
                r_pointer -= 1
                continue

            if s[r_pointer].lower() != s[l_pointer].lower():
                return False
            else:
                l_pointer += 1
                r_pointer -= 1

        return True

input_string = "A man, a plan, a canal: Panama"
print(f'31 Valid Palindrome - {input_string}:'
      f' {Solution().is_palindrome(input_string)}')


""" SORTING ALGOS """

array = [64, 25, 12, 22, 11]

# Selection Sort O(n^2)
# Traverse through all array elements
for i in range(len(array)):

    min_index = i
    for j in range(i + 1, len(array)):
        if array[j] < array[min_index]:
            min_index = j

    array[i], array[min_index] = array[min_index], array[i]

# Driver code to test above
print()
print(f"Selection Sort: Sorted array {array}")

# Bubble Sort
array = [64, 25, 12, 22, 11]

for i in range(len(array)):
    swapped = False
    for j in range(0, len(array) - i - 1):

        if array[j] > array[j+1]:
            array[j], array[j+1] = array[j+1], array[j]
            swapped = True

    if not swapped:
        break

print(f"Bubble Sort: Sorted array {array}")


# Insertion Sort  -  Complexity O(n^2)  - like a stack of cards
array = [64, 25, 12, 22, 11]

for i in range(1, len(array)):
    key = array[i]

    j = i-1
    while key < array[j] and j >= 0:
        array[j+1] = array[j]
        j -= 1

    array[j+1] = key

print(f"Insertion Sort: Sorted array {array}")


class MergeSort:
    def sort_array(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]

            self.sort_array(left)
            self.sort_array(right)

            #Merge
            i = j = k = 0

            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1

                else:
                    arr[k] = right[j]
                    j += 1

                k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

        return arr

arr = [12, 11, 13, 5, 6, 7]
print(f"Merge Sort: Sorted array {arr}: {MergeSort().sort_array(arr)}")

"""
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.right = TreeNode(7)
root.right.left = TreeNode(15)


class SolutionLevel:

    def level_order2(self, node, traversal, level):
        if node is None:
            return
        if len(traversal) <= level:
            traversal.append([])

        traversal[level].append(node.val)

        self.level_order2(node.left, traversal, level+1)
        self.level_order2(node.right, traversal, level+1)

    def level_order(self, node, traversal, level):

        if not node:
            return

        if len(traversal) <= level:
            traversal.append([])

        traversal[level].append(node.val)

        self.level_order(node.left, traversal, level+1)
        self.level_order(node.right, traversal, level+1)

    def levelOrder(self, root: 'TreeNode') -> 'List[List[int]]':
        traversal = []
        self.level_order(root, traversal, 0)
        return traversal


binary_tree = [3, 9, 20, None, None, 15, 7]
print()
print(f'32 Print levels in tree:'
      f' {SolutionLevel().levelOrder(root)}')


class Graph:
    def __init__(self):
        self.graph = collections.defaultdict(list)

    def add_edge(self, u, v):
        if v not in self.graph[u]:
            self.graph[u].append(v)

    def BFS(self, s):
        # mark all the vertices as non-visited
        visited = [False] * (len(self.graph))

        # create a queue for BFS

        queue = []

        # mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:
            # dequeue a vertex from queue and print it
            s = queue.pop(0)
            print(s, end=" ")

            # get all adjacent vertices of the dequeued vertex s
            # if adjacent is not visited - mark it visited and enqueue
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True


g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print(': BFS Traversal from a given source vertex', g.BFS(2))


class Stack:
    def __init__(self):
        self.stack = []

    def pop(self):
        if self.stack:
            return self.stack.pop()

    def push(self, el):
        self.stack.append(el)

    def is_empty(self):
        return self.stack == []


class Queue:
    def __init__(self):
        self.stack1 = Stack()
        self.stack2 = Stack()

    def enqueue(self, item):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())

        self.stack1.push(item)

        while not self.stack2.is_empty():
            self.stack1.push(self.stack2.pop())

    def dequeue(self):
        if not self.stack1.is_empty():
            return self.stack1.pop()


s = Queue()
(s.enqueue(1))
(s.enqueue(2))
(s.enqueue(3))
(s.enqueue(4))
(s.enqueue(5))

print('Implement Queue with 2 Stacks: ', s.dequeue(), s.dequeue(), s.dequeue(), s.dequeue())

""" Binary Search Trees:
Binary Search Tree is a node-based binary tree data structure which has the following properties:
The left subtree of a node contains only nodes with keys lesser than the node’s key.
The right subtree of a node contains only nodes with keys greater than the node’s key.
The left and right subtree each must also be a binary search tree.
"""

"""
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self


class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root = TreeNode(key,val)
        self.size = self.size + 1

    def _put(self,key,val,currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                   self._put(key,val,currentNode.leftChild)
            else:
                   currentNode.leftChild = TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                   self._put(key,val,currentNode.rightChild)
            else:
                   currentNode.rightChild = TreeNode(key,val,parent=currentNode)

    def __setitem__(self,k,v):
       self.put(k,v)

    def get(self,key):
       if self.root:
           res = self._get(key,self.root)
           if res:
                  return res.payload
           else:
                  return None
       else:
           return None

    def _get(self,key,currentNode):
       if not currentNode:
           return None
       elif currentNode.key == key:
           return currentNode
       elif key < currentNode.key:
           return self._get(key,currentNode.leftChild)
       else:
           return self._get(key,currentNode.rightChild)

    def __getitem__(self,key):
       return self.get(key)

    def __contains__(self,key):
       if self._get(key,self.root):
           return True
       else:
           return False

    def delete(self,key):
      if self.size > 1:
         nodeToRemove = self._get(key,self.root)
         if nodeToRemove:
             self.remove(nodeToRemove)
             self.size = self.size-1
         else:
             raise KeyError('Error, key not in tree')
      elif self.size == 1 and self.root.key == key:
         self.root = None
         self.size = self.size - 1
      else:
         raise KeyError('Error, key not in tree')

    def __delitem__(self,key):
       self.delete(key)

    def spliceOut(self):
       if self.isLeaf():
           if self.isLeftChild():
                  self.parent.leftChild = None
           else:
                  self.parent.rightChild = None
       elif self.hasAnyChildren():
           if self.hasLeftChild():
                  if self.isLeftChild():
                     self.parent.leftChild = self.leftChild
                  else:
                     self.parent.rightChild = self.leftChild
                  self.leftChild.parent = self.parent
           else:
                  if self.isLeftChild():
                     self.parent.leftChild = self.rightChild
                  else:
                     self.parent.rightChild = self.rightChild
                  self.rightChild.parent = self.parent

    def findSuccessor(self):
      succ = None
      if self.hasRightChild():
          succ = self.rightChild.findMin()
      else:
          if self.parent:
                 if self.isLeftChild():
                     succ = self.parent
                 else:
                     self.parent.rightChild = None
                     succ = self.parent.findSuccessor()
                     self.parent.rightChild = self
      return succ

    def findMin(self):
      current = self
      while current.hasLeftChild():
          current = current.leftChild
      return current

    def remove(self,currentNode):
         if currentNode.isLeaf(): #leaf
           if currentNode == currentNode.parent.leftChild:
               currentNode.parent.leftChild = None
           else:
               currentNode.parent.rightChild = None
         elif currentNode.hasBothChildren(): #interior
           succ = currentNode.findSuccessor()
           succ.spliceOut()
           currentNode.key = succ.key
           currentNode.payload = succ.payload

         else: # this node has one child
           if currentNode.hasLeftChild():
             if currentNode.isLeftChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.leftChild
             elif currentNode.isRightChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.leftChild
             else:
                 currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
           else:
             if currentNode.isLeftChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.rightChild
             elif currentNode.isRightChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.rightChild
             else:
                 currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)




mytree = BinarySearchTree()
mytree[3]="red"
mytree[4]="blue"
mytree[6]="yellow"
mytree[2]="at"

print(mytree[6])
print(mytree[2])

"""

""" Graph Knight Problem"""


class Vertex:
    def __init__(self, key):
        self.id = key
        self.color = 'white'
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


def posToNodeId(row, col, bdSize):
    return bdSize * row + col

def legalCoord(x, bdSize):
    if x < bdSize and x >= 0:
        return True
    return False

def genLegalMoves(row, col, bdSize):
    newMoves = []
    moveOffsets = [(-1, -2), (-1, 2), (-2, -1), (-2, 1),
                   (1, -2), (1, 2), (2, -1), (2, 1)]

    for a_move in moveOffsets:
        new_x = row + a_move[0]
        new_y = col + a_move[1]

        if legalCoord(new_x, bdSize) and legalCoord(new_y, bdSize):
            newMoves.append(new_x, new_y)

    return newMoves

def knightGraph(bdSize):
    ktGraph = Graph()
    for row in range(bdSize):
        for col in range(bdSize):
            nodeId = posToNodeId(row, col, bdSize)
            newPositions = genLegalMoves(row, col, bdSize)
            for e in newPositions:
                nid = posToNodeId(e[0], e[1], bdSize)
                ktGraph.addEdge(nodeId, nid)

    return ktGraph


def knightTour(n, path, u, limit):
    u.setColor('gray')
    path.append(u)

    if n < limit:
        nbrs_list = list(u.getConnections())
        i = 0
        done = False

        while i < len(nbrs_list) and not done:
            if nbrs_list[i].getColor() == 'white':
                done = knightTour(n + 1, path, nbrs_list[i], limit)
            i += 1

        # backtracking
        if not done:
            path.pop()
            u.setColor('white')

    else:
        done = True
    return done


""" Fibonacci - Exponential Solution(recursion) """
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(f'Fibonacci of 9, recursive: {fibonacci(9)}')


""" Fibonacci - Method 2: Dynamic Programming - store intermediate results """
def fibonacci(n):
    fib_array = [0, 1]

    while len(fib_array) < n + 1:
        fib_array.append(0)

    if n <= 1:
        return n

    if fib_array[n-1] == 0:
        fib_array[n-1] = fibonacci(n-1)

    if fib_array[n-2] == 0:
        fib_array[n-2] = fibonacci(n-2)

    fib_array[n] = fib_array[n-1] + fib_array[n-2]

    return fib_array[n]

print(f'Fibonacci of 9, dynamic: {fibonacci(9)}')


"""Method 2: Dynamic Programming - store only last 2 numbers
    Time complexity - O(n), space - O(1)
"""
def fibonacci(n):
    a = 0
    b = 1
    if n < 0:
        return 'incorrect input'
    elif n == 0:
        return a
    elif n == 1:
        return b

    else:
        for i in range(2, n + 1):
            c = a + b
            a = b
            b = c

        return b


"""
141. Linked List Cycle
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed)
in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.

"""
llist = LinkedList()
llist.insert_item(ListNode('e'))
llist.insert_item(ListNode('d'))
cnode = ListNode('c')
llist.insert_item(cnode)
llist.insert_item(ListNode('b'))
llist.insert_item(ListNode('a'))
llist.insert_item(cnode)


class Solution(object):
    def has_cycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        current = head
        nodes_seen = set()

        while current:
            if current in nodes_seen:
                return True
            nodes_seen.add(current)
            current = current.next

        return False


print(f'33 Linked list has cycle:'
      f' {Solution().has_cycle(llist.head)}')

"""
Given a string S and a string T, find the minimum window in S
which will contain all the characters in T in complexity O(n).

Example:
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
Note:

If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.
"""


class Solution:
    def min_window(self, s: 'str', t: 'str') -> 'str':
        if not s or not t:
            return ''

        t_char_counter = collections.Counter(t)
        required = len(t_char_counter)

        windowcounts = {}

        # pointers
        l = 0
        r = 0

        # this keeps track of how many characters are met in their desired frequency
        formed = 0

        # answer tuple of the form: length, left, right
        ans = (500, 0, 0)

        while r < len(s):
            char = s[r]
            windowcounts[char] = windowcounts.get(char, 0) + 1
            if char in t_char_counter and windowcounts[char] == t_char_counter[char]:
                formed += 1

            # contract the window
            while l <= r and formed == required:
                character = s[l]

                # save smallest window until now
                if r - l + 1 < ans[0]:
                    ans = (r -l + 1, l, r)

                # remove char at the left pointer
                windowcounts[character] -= 1
                if character in t_char_counter and windowcounts[character] < t_char_counter[character]:
                    formed -= 1
                l += 1

            r += 1
        return "" if ans[0] == 500 else s[ans[1]: ans[2]+1]

s = "ADOBECODEBANC"
t = "ABC"

print(f'34 Minimum window substring {t} in {s}: {Solution().min_window(s, t)}')

"""
Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
"""
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]


class Solution:
    def group_anagrams(self, strs):
        ans = collections.defaultdict(list)

        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()

print(f'35 Group anagrams {strs} : {Solution().group_anagrams(strs)}')

"""
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

"""

s = '()[]{}>>['
s = '{[]}'


class Solution:
    def is_valid(self, s: str) -> bool:
        paren_map = {')': '(',
                     '}': '{',
                     ']': '[',
                     '>': '<'
                     }
        stack = []

        for ch in s:
            if ch in paren_map:

                if not len(stack):
                    return False

                match = stack.pop()
                if match != paren_map[ch]:
                    return False
            else:
                # the if statement bulletproofs for other chars by dismissing them
                if ch in paren_map.values():
                    stack.append(ch)

        return len(stack) == 0

print(f'36 Valid parentheses {s} : {Solution().is_valid(s)}')

"""
"Count Palindromes"

Solution: Expand around center
Let N be the length of the string.
The middle of the palindrome could be in one of 2N - 1 positions: either at letter or between two letters.

For each center, let's count all the palindromes that have this center.
Notice that if [a, b] is a palindromic interval (meaning S[a], S[a+1], ..., S[b] is a palindrome),
then [a+1, b-1] is one, too.

Algorithm

For each possible palindrome center, let's expand our
candidate palindrome on the interval [left, right] as long as we can.
The condition for expanding is left >= 0 and right < N and S[left] == S[right].
That means we want to count a new palindrome S[left], S[left+1], ..., S[right].

"""
s = 'abc'
s = 'aaa'


class Solution:
    def count_palindromes(self, s: str) -> int:
        N = len(s)
        result = 0

        for center in range(2 * N - 1):
            left = center // 2
            right = left + center % 2

            while left >= 0 and right < N and s[left] == s[right]:
                result += 1
                left -= 1
                right += 1

        return result

print(f'37 Count valid substring palindromes: {s} : {Solution().count_palindromes(s)}')

"""
TREES

Notes
A tree is an undirected and connected acyclic graph.

Recursion is a common approach for trees. When you notice that the subtree problem can be used to solve the entire problem, try using recursion.

When using recursion, always remember to check for the base case, usually where the node is null.

When you are asked to traverse a tree by level, use depth first search.

Sometimes it is possible that your recursive function needs to return two values.

If the question involves summation of nodes along the way, be sure to check whether nodes can be negative.

You should be very familiar with writing pre-order, in-order, and post-order
traversal recursively. As an extension, challenge yourself by writing them iteratively. Sometimes interviewers ask candidates for the iterative approach
"""
"""
In-order: Left, root, right. For BST - gives nodes in non-decreasing order

Post-order: Root, left, right
Preorder traversal is used to create a copy of the tree. Preorder traversal is also used to get prefix expression on of an expression tree.
"""

"""
Given a binary tree, find its maximum depth.
The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
Note: A leaf is a node with no children.
Example: Given binary tree [3,9,20,null,null,15,7], depth is 3

"""

s = [3,9,20,None,None,15,7]


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

troot = TreeNode(3)
troot.left = TreeNode(9)
troot.right = TreeNode(20)
troot.right.left = TreeNode(15)
troot.right.right = TreeNode(7)
# troot.right.left.left = TreeNode(14)
# troot.right.left.right = TreeNode(16)


class Solution:
    def max_depth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        depth_left = self.max_depth(root.left)
        depth_right = self.max_depth(root.right)

        depth = max(depth_left, depth_right) + 1

        return depth

print(f'38 Max Depth of a Tree is : {s} : {Solution().max_depth(troot)}')


""" Find Identical Trees
Input:     1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

Output: true
Example 2:

Input:     1         1
          /           \
         2             2

        [1,2],     [1,null,2]

Output: false
Example 3:

Input:     1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

Output: false

"""

root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20)


root2 = TreeNode(3)
root2.left = TreeNode(9)
root2.right = TreeNode(20)


class Solution:
    def identical_trees(self, root1: TreeNode, root2: TreeNode) -> bool:
        # both nodes are None
        if root1 is None and root2 is None:
            return True

        # both nodes exist
        elif root1 is not None and root2 is not None:
            return (root1.val == root2.val
                    and self.identical_trees(root1.left, root2.left)
                    and self.identical_trees(root1.right, root2.right)
                    )

        # only one node is not None
        else:
            return False


print(f'39 Trees are identical : {Solution().identical_trees(root1, root2)}')

"""
Invert a binary tree: Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
Trivia:
This problem was inspired by this original tweet by Max Howell:

Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.
"""
root1 = TreeNode(4)
root1.left = TreeNode(2)
root1.right = TreeNode(7)
root1.right.right = TreeNode(9)
root1.right.left = TreeNode(6)
root1.left.right = TreeNode(3)
root1.left.left = TreeNode(1)

root2 = TreeNode(4)
root2.left = TreeNode(2)
root2.right = TreeNode(7)
root2.right.right = TreeNode(9)
root2.right.left = TreeNode(6)
root2.left.right = TreeNode(3)
root2.left.left = TreeNode(1)


def print_bst(root):
    return (SolutionLevel().levelOrder(root))


class Solution:
    # inverts the entire tree
    def invert_tree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None

        temp = root.left
        root.left = self.invert_tree(root.right)
        root.right = self.invert_tree(temp)

        return root

    # inverts only values of children nodes
    def invert_tree2(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None

        if root.right is None and root.left is None:
            return root

        else:
            root.left.val, root.right.val = root.right.val, root.left.val

            root.left = self.invert_tree2(root.left)
            root.right = self.invert_tree2(root.right)

        return root


print(f'40 Invert Binary Tree - Complete Inverse: {print_bst(root1)} : {print_bst(Solution().invert_tree(root1))}')
print(f'40.1 Invert Binary Tree - Values Only: {print_bst(root2)} : {print_bst(Solution().invert_tree2(root2))}')

## Find max sum leaf to root in a binary tree:

root = TreeNode(10)
root.left = TreeNode(-2)
root.right = TreeNode(7)
root.left.right = TreeNode(-4)
root.left.left = TreeNode(8)


class Solution:
    def print_leaf_to_root(self, root: TreeNode, target_leaf: TreeNode, path: 'List[int]') -> bool:
        if root is None:
            return False

        if (target_leaf.val == root.val
            or self.print_leaf_to_root(root.left, target_leaf, path)
            or self.print_leaf_to_root(root.right, target_leaf, path)):

            path.insert(0, root.val)

            return True

        return False

    def get_target_leaf(self, current_sum, current_node, ans):

        # not a leaf node
        if current_node is None:
            return

        current_sum = current_sum + current_node.val

        # leaf node
        if current_node.left is None and current_node.right is None:
            if current_sum > ans['max_sum']:
                ans['max_sum'] = current_sum
                ans['max_node'] = current_node

        else:
            self.get_target_leaf(current_sum, current_node.left, ans)
            self.get_target_leaf(current_sum, current_node.right, ans)

    def max_leaf_to_node(self, root: TreeNode) -> int:
        if root is None:
            return None

        answer = {'max_sum': root.val, 'max_node': root}

        self.get_target_leaf(0, root, answer)

        path = []
        self.print_leaf_to_root(root, answer['max_node'], path)
        return path


print(f'41 Max Leaf to Root in Binary Tree: {print_bst(root)} : {Solution().max_leaf_to_node(root)}')

###Maximum Path Sum - 124
"""
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along
the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42
"""
"""
We can find the maximum sum using single traversal of binary tree. The idea is to maintain two values in recursive calls
1) Maximum root to leaf path sum for the subtree rooted under current node.
2) The maximum path sum between leaves (desired output).

For every visited node X, we find the maximum root to leaf sum in left and right subtrees of X.
We add the two values with X->data, and compare the sum with maximum path sum found so far.

Following is the implementation of the above O(n) solution.
"""
# Utility function to find maximum sum between any
# two leaves. This function calculates two values:
# 1) Maximum path sum between two leaves which are stored
#    in res
# 2) The maximum root to leaf path sum which is returned
# If one side of root is empty, then it returns INT_MIN

root = TreeNode(-10)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)


def max_path_sum_util(root, res):
    # Base Case
    if root is None:
        return 0

    if root.left is None and root.right is None:
        return root.val

    # Find max sum in left and right subtree. Also
    # find maximum root to leaf sums in left and right
    # subtrees and store them in ls and rs
    ls = max_path_sum_util(root.left, res)
    rs = max_path_sum_util(root.right, res)

    # If both left and right children exist
    if root.left is not None and root.right is not None:

        # update result if needed
        res[0] = max(res[0], ls + rs + root.val)

        # Return maximum possible value for root being
        # on one side
        return max(ls, rs) + root.val

    # If any of the two children is empty, return
    # root sum for root being on one side
    if root.left is None:
        return rs + root.val
    else:
        return ls + root.val


# The main function which returns sum of the maximum
# sum path between two leaves. THis function mainly
# uses maxPathSumUtil()
def max_path_sum(root):
    res = [-2**32]
    max_path_sum_util(root, res)
    return res[0]

print(f'42 Max Path Sum Between Any Nodes in Binary Tree: {print_bst(root)} : {max_path_sum(root)}')

"""
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored
in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work.
You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Example:

You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(5)


class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        vals = []

        def encode(node):
            if node:
                vals.append(str(node.val))
                encode(node.left)
                encode(node.right)
            else:
                vals.append('#')

        encode(root)

        return ' '.join(vals)

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def decode(vals):
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = decode(vals)
            node.right = decode(vals)
            return node

        vals = iter(data.split())
        return decode(vals)

        # Your Codec object will be instantiated and called as such:
        # codec = Codec()
        # codec.deserialize(codec.serialize(root))

print(f'43 Serialize/Deserialize Binary Tree: {print_bst(root)} : '
      f'{Codec().serialize(root)}:'
      f'{print_bst(Codec().deserialize(Codec().serialize(root)))}')


class Stack(object):
    def __init__(self):
        """Initialize an empty stack"""
        self.items = []

    def push(self, item):
        """Push a new item onto the stack"""
        self.items.append(item)

    def pop(self):
        """Remove and return the last item"""
        # If the stack is empty, return None
        # (it would also be reasonable to throw an exception)
        if not self.items:
            return None

        return self.items.pop()

    def peek(self):
        """Return the last item without removing it"""
        if not self.items:
            return None
        return self.items[-1]


llist = LinkedList()
llist.insert_item(ListNode('d'))
llist.insert_item(ListNode('c'))
it = llist.insert_item(ListNode('b'))
llist.insert_item(ListNode('a'))


print(f'44 Delete Node from Singly Linked List With only reference to the node: {(llist.print_list())}: deleting {it.val}')
# move data from next node to the current node
# delete next node
def delete_node(node):
    node.val = node.next.val
    node.next = node.next.next

delete_node(it)
print(f'44 Delete Node from Singly Linked List With only reference to the node: {(llist.print_list())}')


llist = LinkedList()
llist.insert_item(ListNode('f'))
llist.insert_item(ListNode('e'))
llist.insert_item(ListNode('d'))
llist.insert_item(ListNode('c'))
it = llist.insert_item(ListNode('b'))
llist.insert_item(ListNode('a'))


""" Kth element from the end of the list """
class Solution:
    def kth_element(self, k, head: 'LinkedList') -> int:
        trailing = current = head

        i = 0
        while i < k and current is not None:
            current = current.next
            i += 1

        # list is shorter than k elements
        if i != k:
            raise Exception(f'List shorter than {k} elements')

        while current is not None:
            current = current.next
            trailing = trailing.next

        return trailing.val

print(f'45 Kth (3) Element from the End in a Linked List {print_ll(llist.head)}'
      f'is {(Solution().kth_element(3, llist.head))}')

# 572 Subtree of Another Tree
"""
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure
and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants
The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
"""

root = TreeNode(3)
root.left = TreeNode(4)
root.right = TreeNode(5)
root.left.left = TreeNode(1)
root.left.right = TreeNode(2)

root2 = TreeNode(3)
root2.left = TreeNode(4)
root2.right = TreeNode(5)
root2.left.left = TreeNode(1)
root2.left.right = TreeNode(2)
root2.left.right.left = TreeNode(0)

root_sub = TreeNode(4)
root_sub.left = TreeNode(1)
root_sub.right = TreeNode(2)


"""VV approach:
        1. traverse the first tree until xnode.val == sub_root.val
        2. check if trees with root at xnode and sub_root are identical
        3. if 2 is true -> return; else keep looking
        4. return False
"""


class Solution:
    def trees_identical(self, root1, root2):
        if root1 is None and root2 is None:
            return True

        elif root1 is not None and root2 is not None:
            return (root1.val == root2.val
                    and self.trees_identical(root1.left, root2.left)
                    and self.trees_identical(root1.right, root2.right)
                    )
        else:
            return False

    def is_subtree(self, t: TreeNode, sub: TreeNode) -> bool:
        if sub is None:
            return True

        elif t is not None and sub is not None:
            if self.trees_identical(t, sub):
                return True

            else:
                # could arguably return "self.is_subtree(s.left, t) or self.is_subtree(s.right, t)"
                # this will lead to extra evaluation of right subtree even if left returned true

                if self.is_subtree(t.left, sub):
                    return True
                return self.is_subtree(t.right, sub)

        else:
            return False

print(f'46 Subtree t {print_bst(root_sub)} of another tree s {print_bst(root)} : '
      f'{Solution().is_subtree(root, root_sub)}')
print(f'46.2 Subtree t {print_bst(root_sub)} of another tree s {print_bst(root2)} : {print_bst(root)} : '
      f'{Solution().is_subtree(root2, root_sub)}')


### Binary Tree Traversals:
# Auxiliary Space : If we don’t consider size of stack for function calls then O(1) otherwise O(n).

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

    # A function to do inorder tree traversal
def printInorder(root):

    if root:

        # First recur on left child
        printInorder(root.left)

        # then print the data of node
        print(root.val),

        # now recur on right child
        printInorder(root.right)


# A function to do postorder tree traversal
def printPostorder(root, trav=[]):

    # if root:
    #
        # First recur on left child
        # printPostorder(root.left, trav)
        #
        # the recur on right child
        # printPostorder(root.right, trav)
        #
        # now print the data of node
        # trav.append(root.val)
    #
    # return trav
    if root:
        printPostorder(root.left)
        printPostorder(root.right)
        print(root.val)

    # A function to do preorder tree traversal
def printPreorder(root):

    if root:

        # First print the data of node
        print(root.val),

        # Then recur on left child
        printPreorder(root.left)

        # Finally recur on right child
        printPreorder(root.right)


# Driver code
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
print("Preorder traversal of binary tree is")
printPreorder(root)

print("\nInorder traversal of binary tree is")
printInorder(root)

print(f"Postorder traversal of binary tree is")
printPostorder(root, [])

"""
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
"""

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    """
    Brute force: traverse all linked lists - put them in an array
    sort array
    compose linked list with elements from the array
    Time Complexity: O(nlogn)
    """
    def mergeKLists(self, lists: []) -> LinkedList:
        master_list = []

        # Get all vals in array
        for a_list in lists:
           current_node = a_list

           while current_node:
                master_list.append(current_node.val)
                current_node = current_node.next

        master_list_sorted = sorted(master_list)
        last_node = ListNode(master_list_sorted[0])
        current = head = last_node

        for a_val in master_list_sorted[1:]:
            new_node = ListNode(a_val)
            current.next = new_node
            current = current.next

        return head

    """
    Merge two lists k-1 times
    Time Complexity: O(Nk)
    Space Complexity: O(1)
    """
    def merge_k_lists(self, lists: []) -> LinkedList:
        def compare_two_lists(l1, l2):
            if l1.val < l2.val:
                current = head = l1
                l1 = l1.next
            else:
                current = head = l2
                l2 = l2.next

            while l1 is not None and l2 is not None:
                if l1.val > l2.val:
                    current.next, l2 = l2, l2.next
                else:
                    current.next, l1 = l1, l1.next

                current = current.next

            if l1 is not None:
                current.next = l1
                # while l1 is not None:
                    # current.next, l1 = l1, l1.next
                    # current = current.next

            if l2 is not None:
                current.next = l2
                # while l2 is not None:
                    # current.next, l2 = l2, l2.next
                    # current = current.next

            return head

        final_list = lists[0]
        i = 1
        while i < len(lists):
            final_list = compare_two_lists(final_list, lists[i])
            i += 1

        return final_list


l1 = ListNode(1)
l1.next = ListNode(4)
l1.next.next = ListNode(5)

l2 = ListNode(1)
l2.next = ListNode(3)
l2.next.next = ListNode(4)

l3 = ListNode(2)
l3.next = ListNode(6)

llists = [l1, l2, l3]

def print_llist_from_head(head):
    outcome = []
    while head:
        outcome.append(str(head.val))
        head = head.next
    return ', '.join(outcome)

print()
print('47: Merge K sorted linked lists: ', print_llist_from_head(Solution().merge_k_lists(llists)))

"""
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:

Given 1->2->3->4, reorder it to 1->4->2->3.
Example 2:

Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
Algo:
1. Get the pointer to the middle of the llist
2. Reverse the second part of the list
3. Merge first and second halves
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

l = ListNode(1)
l.next = ListNode(2)
l.next.next = ListNode(3)
l.next.next.next = ListNode(4)
l.next.next.next.next = ListNode(5)


class Solution:
    def reorder_list2(self, head: 'ListNode') -> 'None':
        if head is None:
            return
        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        current = slow
        previous = None

        while current:
            temp = current.next
            current.next = previous
            previous = current
            current = temp

        first = head
        second = previous

        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next

        return head

print('48: Reorder list L0->LN->L1->LN-1-> : ', print_llist_from_head(Solution().reorder_list2(l)))

"""
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
"""
Algo:
1. pick element from preorder :increase pindex
2. create a  newNode with val
3. find picked element's index in inorder
4. call buildtree for elements before inIndex as left subtree of 2
4.2 call buildtree for elements after inIndex as right subtree of 2
5. return newNode

"""

preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]


class Solution:
    pindex = 0
    # with a class var
    def build_tree(self, preorder: 'List[int]', inorder: 'List[int]') -> TreeNode:
        """
        :param preorder:
        :param inorder:
        :return:
        """
        if len(inorder) == 1:
            self.pindex += 1
            return TreeNode(inorder[0])

        new_val = preorder[self.pindex]
        new_node = TreeNode(new_val)

        in_index = inorder.index(new_val)

        self.pindex += 1
        new_node.left = self.build_tree(preorder, inorder[:in_index])
        if in_index < len(inorder) - 1:
            new_node.right = self.build_tree(preorder, inorder[in_index+1:])

        return new_node

    # without a class var - iterator implementation
    def build_tree2(self, preorder_it: 'List[int]', inorder: 'List[int]') -> TreeNode:
        """
        :param preorder:
        :param inorder:
        :return:
        """
        if len(inorder) == 1:
            next(preorder_it)
            return TreeNode(inorder[0])

        new_val = next(preorder_it)
        new_node = TreeNode(new_val)

        in_index = inorder.index(new_val)

        new_node.left = self.build_tree2(preorder_it, inorder[:in_index])
        new_node.right = self.build_tree2(preorder_it, inorder[in_index+1:])

        return new_node


def serialize_bst(root):
    vals = []

    def encode(root):
        if root:
            vals.append(str(root.val))
            encode(root.left)
            encode(root.right)

        else:
            vals.append('#')

    encode(root)
    return ', '.join(vals)

print('49: Build tree from pre-order and post-order lists of elements: ', serialize_bst((Solution().build_tree(preorder,
                                                                                                               inorder))))

preorder_it = iter(preorder)
print('49: Build tree from pre-order and post-order lists of elements: ', serialize_bst((Solution().build_tree2(preorder_it,
                                                                                                               inorder))))
"""
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not
a multiple of k then left-out nodes in the end should remain as it is.
Example:

Given this linked list: 1->2->3->4->5

For k = 2, you should return: 2->1->4->3->5

For k = 3, you should return: 3->2->1->4->5

Note:

Only constant extra memory is allowed.
You may not alter the values in the list's nodes, only nodes itself may be changed.

def reverse_list(head, m, n):
    current = head
    previous = None

    left_pointer = 0
    reversed_items = n - m + 1

    while current is not None:
        while left_pointer < m - 1:
            previous = current
            current = current.next
            left_pointer += 1
        # from where to join to the next element
        left_current = previous

        # from where to join to the rest of the list
        right_current = current

        for i in range(reversed_items):
            temp = current.next
            current.next = previous
            previous = current
            current = temp

        left_current.next = previous
        right_current.next = current

        return head

Algo:
1. Utilize reverse func between m and n elements
2. Count elements in linked list
3. Break up LL into chunks and reverse each chunk separately

more efficient is to advance the pointer
"""

def reverse_list(head, m, n):
    current = head
    previous = None

    left_pointer = 0
    reversed_items = n - m + 1

    while current is not None:
        while left_pointer < m - 1:
            previous = current
            current = current.next
            left_pointer += 1
        # from where to join to the next element
        left_current = previous

        # from where to join to the rest of the list
        right_current = current

        for i in range(reversed_items):
            temp = current.next
            current.next = previous
            previous = current
            current = temp

        if left_current is not None:
            left_current.next = previous
        else:
            head = previous
        right_current.next = current

        return head


def reverse_k_batches(head, k):
    items_count = 0
    current = head
    while current:
        items_count += 1
        current = current.next

    if k < 1:
        return head
    elif items_count < k:
        return head
    else:
        no_batches = items_count // k
        # immutable_items = items_count % k

        reverse_batches = [[k*i+1, k*i+k] for i in range(no_batches)]

        for a_batch in reverse_batches:
            head = reverse_list(head, a_batch[0], a_batch[1])

    return head


llist = LinkedList()
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))

print('50: Reverse the nodes k=3 elements at a time: ', print_ll(reverse_k_batches(llist.head, 3)))

"""
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.
"""
class Solution:
    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
        if n < 1:
            raise Exception('Please provide positive n')

        #keep the tracking node behind deletion target
        tracking = current = head

        for i in range(n + 1):

            # edge case: deleting the first element
            if not current:
                return head.next

            current = current.next

        while current:
            tracking = tracking.next
            current = current.next

        tracking.next = tracking.next.next

        #delete tracking node by deleting the next node and moving next node's data to tracking
        # print (tracking,current)
        # if tracking.next is None:
        #     tracking = None
        # else:
        #     tracking.val = tracking.next.val
        #     tracking.next = tracking.next.next

        return head

llist = LinkedList()
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))

print('50: Remove nth=3 node from the end of linked list: ', print_ll(Solution().remove_nth_from_end(llist.head, 3)))

"""
Remove all elements from a linked list of integers that have value val.

Example:

Input:  1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5
"""
llist = LinkedList()
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))


class Solution:
    def remove_elements(self, head: ListNode, n: int) -> ListNode:
        current = head

        # Deal with the case where first node contains target val
        while current and current.val == n:
            current = current.next
            head = current

        if current:
            one_plus = current.next
        else:
            raise Exception('nothing to return - linked list only contains target vals')

        while one_plus:
            if one_plus.val == n:
                current.next = current.next.next
                one_plus = current.next
            else:
                current = current.next
                one_plus = one_plus.next

        return head

print('51: Remove nodes from linked list with value 6: ', print_ll(Solution().remove_elements(llist.head, 6)))
"""
Given a sorted linked list, delete all nodes that have duplicate numbers,
leaving only distinct numbers from the original list.

Example 1:

Input: 1->2->3->3->4->4->5
Output: 1->2->5
Example 2:

Input: 1->1->1->2->3
Output: 2->3
"""


class Solution:
    def remove_duplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        # when duplicates appear right away
        if head.next and head.val == head.next.val:
            del_val = head.val
            while head and head.val == del_val:
                head = head.next

        # general case
        tracking, current = head, head.next

        while current and current.next:
            if current.val == current.next.val:
                while current.next and current.val == current.next.val:
                    current = current.next
                current = current.next

            else:
                tracking.next = current
                tracking = tracking.next
                current = current.next

        if tracking.next != current:
            tracking.next = None

        return head

llist = LinkedList()

llist.insert_item(ListNode(7))
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))

llist2 = LinkedList()
llist2.insert_item(ListNode(3))
llist2.insert_item(ListNode(2))
llist2.insert_item(ListNode(1))
llist2.insert_item(ListNode(1))
llist2.insert_item(ListNode(1))

print(f'52: Remove duplicates from linked list {print_ll(llist.head)}: ', print_ll(Solution().remove_duplicates(llist.head)))
print(f'52: Remove duplicates from linked list {print_ll(llist2.head)}: ', print_ll(Solution().remove_duplicates(llist2.head)))


"""
Write a program to find the node at which the intersection of two singly linked lists begins.
listA = [0,9,1,2,4], listB = [3,2,4]

"""


# Basic idea is that at some point they will probably intersect same node distance from the end.
# As long as we start traversing at the same distance from the end, we can check equality of nodes
class Solution:
    def find_intersection(self, head1: ListNode, head2: ListNode) -> ListNode:
        len1 = len2 = 0

        # find len1
        current = head1
        while current:
            current = current.next
            len1 += 1

        # find len2
        current = head2
        while current:
            current = current.next
            len2 += 1

        current1 = head1
        current2 = head2

        if len1 > len2:
            for i in range(len1 - len2):
                current1 = current1.next

        else:
            for i in range(len2 - len1):
                current2 = current2.next

        while current1 and current2:
            if current1.val == current2.val:
                return current1

            current1 = current1.next
            current2 = current2.next

        return None


def make_linked_list_from_list(li: list) -> LinkedList:
    """
        Helper function to create a linked list from an array of values
    :param li: list with values for nodes of LinkedList
    :return: LinkedList with items from the list
    """

    llist = LinkedList()
    for an_item in li[::-1]:
        llist.insert_item(ListNode(an_item))
    return llist

listA = [0, 9, 1, 2, 4]
listB = [3, 2, 4]
llist = make_linked_list_from_list(listA)
llist2 = make_linked_list_from_list(listB)

rt_val = Solution().find_intersection(llist.head, llist2.head)
print(f'53: Find intersection of lists {print_ll(llist.head)} and {print_ll(llist2.head)}: ', rt_val.val if rt_val is not None else 'None')


"""
Sort a linked list in O(n log n) time using constant space complexity.

Example 1:

Input: 4->2->1->3
Output: 1->2->3->4
Example 2:

Input: -1->5->3->4->0
Output: -1->0->3->4->5

"""

input1 = [4, 2, 1, 3]
input2 = [-1, 5, 3, 4, 0]


class Solution:
    def sort_linked_list(self, input: ListNode) -> ListNode:
        array = []

        current = input
        while current:
            array.append(current.val)
            current = current.next

        sorted_arr = sorted(array)

        sorted_ll = LinkedList()
        for an_element in sorted_arr[::-1]:
            sorted_ll.insert_item(ListNode(an_element))

        return sorted_ll.head


ll1 = make_linked_list_from_list(input1)
ll2 = make_linked_list_from_list(input2)

print(f'54: Sort linked list{print_ll(ll1.head)}:', print_ll(Solution().sort_linked_list(ll1.head)))
print(f'54.2: Sort linked list{print_ll(ll2.head)}:', print_ll(Solution().sort_linked_list(ll2.head)))

"""
Algorithm of Insertion Sort:

Insertion sort iterates, consuming one input element each repetition, and growing a sorted output list.
At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there.
It repeats until no input elements remain.

Example 1:

Input: 4->2->1->3
Output: 1->2->3->4
Example 2:

Input: -1->5->3->4->0
Output: -1->0->3->4->5
"""


class Solution:
    def insertion_sort(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        first_it = trailing = head
        current = head.next

        while trailing.next:

            if current.val >= trailing.val:
                trailing = current
                current = current.next

            else:

                trailing.next = current.next
                # the very head of the list
                if current.val < first_it.val:
                    temp = first_it

                    first_it = current
                    first_it.next = temp

                else:
                    it = first_it

                    while it.val < trailing.val:
                        if it.next.val > current.val:
                            temp = it.next
                            it.next = current
                            current.next = temp
                            break
                        else:
                            it = it.next

                current = trailing.next

        return first_it


print(f'55: Sort linked list using insertion sort: {print_ll(ll1.head)}: ', print_ll(Solution().insertion_sort(ll1.head)))
print(f'55.2: Sort linked list using insertion sort: {print_ll(ll2.head)}:', print_ll(Solution().insertion_sort(ll2.head)))


"""
86. Given a linked list and a value x, partition it such that all nodes less
than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

Example:

Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
"""


class Solution:
    """
        1. use two pointers to track lower vals and upper vals
        2. find beginning of left and right part
        3. traverse the list from beginning, appending elements to the appropriate parts of the list

    """
    def partition_llist(self, head: ListNode, pivot: int) -> ListNode:
        left_it = right_it = head

        if head.val < pivot:
            while right_it and right_it.val < pivot:
                right_it = right_it.next
        else:
            while left_it and left_it.val > pivot:
                left_it = left_it.next

        right_current = right_it
        left_current = left_it

        while head:
            if head == left_it or head == right_it:
                head = head.next
                continue

            if head.val < pivot:
                left_current.next = head
                left_current = left_current.next
            else:
                right_current.next = head
                right_current = right_current.next

            head = head.next

        # link left and right sublists, close the right sublist
        left_current.next = right_it
        right_current.next = None

        return left_it


input_list = [1, 4, 3, 2, 5, 2]
pivot = 3

ll = make_linked_list_from_list(input_list)

print(f'56: Partition List: {print_ll(ll.head)} relative to {pivot}: ', print_ll(Solution().partition_llist(ll.head, pivot)))


"""
Given an array of n positive integers and a positive integer s, find the minimal
length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example:

Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n).


# Brute force:
# from each element find the minimal length of array that exceeds the sum
# return this minimal length
Intuition for Memoization: Slightly altered approach - sum in O(1)

In Approach #1, you may notice that the sum is calculated for every subarray in O(n)O(n) time.
But, we could easily find the sum in O(1) time by storing the cumulative sum from the beginning(Memoization).
After we have stored the cumulative sum in sums, we could easily find the sum of any subarray from ii to jj.

Space: O(n), Time: O(n)
"""


class Solution:
    def find_min_subarray(self, nums, s):
        min_length = len(nums)

        for i in range(len(nums)):
            sum_so_far = nums[i]
            if sum_so_far >= s:
                return 1

            for j in range(i + 1, len(nums)):
                sum_so_far += nums[j]

                if sum_so_far >= s:
                    min_length = min(j-i+1, min_length)
                    break

        return min_length if min_length < len(nums) else 0

    def find_min_subarray2(self, nums, s):
        """ Memoization - extra array that contains sums of all elements
            kind of the same thing, plus extra space complexity """
        min_length = len(nums)

        sums = [nums[0]]*min_length

        for i in range(1, min_length):
            sums[i] = sums[i-1] + nums[i]

        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sum_so_far = sums[j] - sums[i] + nums[i]

                if sum_so_far >= s:
                    min_length = min(j-i+1, min_length)
                    break

        return min_length if min_length < len(nums) else 0

    """
    Note that in Approach #2, we search for subarray starting with index i
    until we find sum=sums[j]−sums[i]+nums[i] that is greater than s.
    So, instead of iterating linearly to find the sum, we could use binary search to find the index that is not lower than
    s−sums[i] in the sums, which can be done using lower_bound function in C++ STL or could be implemented manually.
    """

    def find_min_subarray4(self, nums, s):
        """ good old two pointers solution
            time complexity O(n); space complexity O(1)
        """
        ans = len(nums)

        left = 0
        sum = 0

        for i in range(len(nums)):
            sum += nums[i]
            while sum >= s:
                ans = min(ans, i + 1 - left)
                sum -= nums[left]
                left += 1

        return ans if ans != len(nums) else 0

target_sum = 7
nums = [2, 3, 1, 2, 4, 3]
print(f'57: Min subarray of {nums} with sum greater than {target_sum}: ', Solution().find_min_subarray(nums, target_sum))
print(f'57.2 Kindof Same thing: Memoization: Min subarray of {nums} with sum greater than {target_sum}: ',
      Solution().find_min_subarray2(nums, target_sum))
print(f'57.4: Two pointers: Min subarray of {nums} with sum greater than {target_sum}: ', Solution().find_min_subarray4(nums, target_sum))


"""
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.
Given target = 20, return false.

"""


class Solution:
    def search_matrix(self, vals, target):
        """
        Brute force - search everything
        Time complexity: O(M*N)
        """
        for i in range(len(vals)):
            for j in range(len(vals[0])):
                if vals[i][j] == target:
                    return True

        return False

    def search_matrix2(self, vals, target):
        """
            Start with the top right corner:
                - if el < target, move one row ++
                - if el > target, move one col --
                - if el == target, return True

            Time complexity O(M+N)
        """

        row = 0
        col = len(vals[0]) - 1

        while row < len(vals) and col > 0:
            element = vals[row][col]

            if element == target:
                return True
            elif element < target:
                row += 1
            else:
                col -= 1

        return False


vals = \
[
    [1,   4,  7, 11, 15],
    [2,   5,  8, 12, 19],
    [3,   6,  9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
]
target = 20
target2 = 5

print(f'571: Search matrix {vals} for target {target} : ', Solution().search_matrix(vals, target))
print(f'571.2: Search matrix {vals} for target {target2} : ', Solution().search_matrix2(vals, target2))


"""
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check.
Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version.
You should minimize the number of calls to the API.

Example:

Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version.
"""

def is_bad_version(version):
    if version > 3:
        return True

    return False


class Solution:
    def first_bad_version(self, versions):
        left = 0
        right = len(versions)

        while left < right:
            mid = left + (right - left) // 2
            if is_bad_version(mid):
                right = mid
            else:
                left = mid + 1

        return left


versions = [0, 1, 2, 3, 4, 5]
print(f'58: Is bad version in versions {versions}: {Solution().first_bad_version(versions)}')


"""Perfect Square
Given a positive integer num, write a function which returns True if num is a perfect square else False.

Note: Do not use any built-in library function such as sqrt.

Example 1:

Input: 16
Output: true
Example 2:

Input: 14
Output: false
"""


class Solution:
    def is_perfect_square(self, num: int) -> bool:
        """
            :param num:
            :return: True or False
        """
        left = 0
        right = num

        while left < right:
            mid = left + (right - left) // 2
            res = mid * mid
            if res == num:
                return True

            elif res > num:
                right = mid
            else:
                left = mid + 1

        return False

num = 16
num = 14

print(f'59 Perfect square of {num} exists?: {Solution().is_perfect_square(num)}')


"""
Given a n x n matrix where each of the rows and columns are sorted in ascending order,
find the kth smallest element in the matrix.
Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note:
You may assume k is always valid, 1 ≤ k ≤ 2n.

"""
from typing import List


class Solution:
    def kth_smallest_element(self, matrix: List[List[int]], k: int) -> int:
        m = len(matrix)

        lower = matrix[0][0]
        upper = matrix[m-1][m-1]

        while lower < upper:
            mid = lower + ((upper-lower) // 2)
            count = self.find_count(matrix, mid)
            if count < k:
                lower = mid + 1
            else:
                upper = mid

        return upper

    @staticmethod
    def find_count(matrix, target):
        m = len(matrix)
        i = m - 1
        j = 0
        count = 0

        while i >= 0 and j < m:
            if matrix[i][j] <= target:
                count += i + 1
                j += 1
            else:
                i -= 1

        return count

matrix = [
             [1,  5,  9],
             [10, 11, 13],
             [12, 13, 15]
         ]
k = 8

print(f'60 Kth ({k}th) smallest element in sorted matrix {matrix} is {Solution().kth_smallest_element(matrix, k)}')


"""
Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:

Input:
"tree"

Output:
"eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
Example 2:

Input:
"cccaaa"

Output:
"cccaaa"

Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.
"""
import heapq


class Solution:
    def sort_by_frequency1(self, input_string):
        char_dict = collections.defaultdict(int)
        for a_char in input_string:
            char_dict[a_char] += 1

        chars_list = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
        res = ''
        for a_char, chars_num in chars_list:
            res += a_char * chars_num

        return res

    def sort_by_frequency2(self, input_string: str) -> str:

        count = collections.Counter(input_string)
        h = []

        for word, fre in count.items():
            heapq.heappush(h, (fre, word))

        res = ''
        for i in range(len(h)):
            f, w = heapq.heappop(h)
            res = f * w + res

        return res


input_string = 'tree'
print(f'61 Sort characters in {input_string} by frequency: dict solution:', Solution().sort_by_frequency1(input_string))
print(f'61 Sort characters in {input_string} by frequency: heap solution', Solution().sort_by_frequency2(input_string))



"""
Sort a nearly sorted (or K sorted) array
Given an array of n elements, where each element is at most k away from its target position,
devise an algorithm that sorts in O(n log k) time. For example, let us consider k is 2, an element at index 7 in the
sorted array, can be at indexes 5, 6, 7, 8, 9 in the given array.

Examples:

Input : arr[] = {6, 5, 3, 2, 8, 10, 9}
            k = 3
"""

from heapq import heappop, heappush, heapify


class Solution:

    def sort_array(self, arr: list, k: int):
        """
        :param arr: input array
        :param k: max distance, which every
        element is away from its target position.
        :return: sorted array
        """

        n = len(arr)
        # List of first k+1 items
        heap = arr[:k + 1]

        # using heapify to convert list
        # into heap(or min heap)
        heapify(heap)

        # "rem_elmnts_index" is index for remaining
        # elements in arr and "target_index" is
        # target index of for current minimum element
        # in Min Heap "heap".
        target_index = 0
        for rem_elmnts_index in range(k + 1, n):
            arr[target_index] = heappop(heap)
            heappush(heap, arr[rem_elmnts_index])
            target_index += 1

        while heap:
            arr[target_index] = heappop(heap)
            target_index += 1

        return arr

k = 3
input_array = [2, 6, 3, 12, 56, 8]

print(f'62 Sort array {input_array} where elements are at most k places out of order', Solution().
      sort_array(input_array, k))


"""
    Find median of two sorted arrays

"""


class Solution:
    """
        General approach:
            - assume arrays A and B are nonempty sorted arrays
            - the challenge is to find the median without combining/sorting the final array(s)
            - median value is somewhere at the half of the combined sorted array
            - we need to look for contribution of A and/or B to the final array
            - deploy binary search in order to find that contribution amount
            - we compare the last contrib element of A(x) to last contrib of B(y) and the one next to it(y')

    """

    def find_median(self, inputA, inputB):

        lenA = len(inputA)
        lenB = len(inputB)

        median_position = (lenA + lenB) // 2

        # find max contribution from list A
        min_from_A = median_position + 1 - lenB
        max_from_A = min(median_position + 1, lenA)

        while min_from_A <= max_from_A:

            # print('min-max', min_from_A, max_from_A)
            contrib_A = min_from_A + (max_from_A - min_from_A) // 2
            contrib_B = median_position + 1 - contrib_A
            # print('contrib', contrib_A, contrib_B)

            # If x' < y -> increase contribution from A
            if inputA[contrib_A] < inputB[contrib_B - 1]:
                min_from_A = contrib_A + 1

            # If x > y' -> increase contribution from B
            elif inputA[contrib_A - 1] > inputB[contrib_B]:
                max_from_A = contrib_A - 1

            # Else return the max(x, y) if Len(entire array) is odd; return x+y/2 if  even
            else:
                if (len(inputA)+len(inputB)) % 2 == 1:
                    ans = max(inputA[contrib_A - 1], inputB[contrib_B - 1])
                else:
                    ans = (inputA[contrib_A - 1] + inputB[contrib_B - 1])/2
                return ans

            # print('post: ', contrib_A, contrib_B, 'min-max', min_from_A, max_from_A)

    def find_median_index_solution(self, inputA, inputB):
        max_len = len(inputA) + len(inputB)
        mid_len = max_len // 2

        minA = max(0, mid_len - len(inputB))
        maxA = min(len(inputA), mid_len)

        while minA <= maxA:
            mid = minA + (maxA - minA) // 2
            if inputA[mid] > inputB[mid_len - mid]:
                maxA = mid - 1
            elif inputA[mid + 1] < inputB[mid_len - mid - 1]:
                minA = mid + 1

            else:
                if max_len % 2 == 1:
                    return max(inputA[mid], inputB[mid_len - mid - 1])
                return (inputA[mid] + inputB[mid_len - mid - 1]) / 2


input_array1 = [4, 20, 32, 50, 55, 61]
input_array2 = [1, 15, 22, 30, 70, 80]

input_array3 = [1, 3, 5]
input_array4 = [2, 4, 6]

print(f'63 Find median of two arrays: {input_array1} and {input_array2} in log time:',
      Solution().find_median(input_array1, input_array2))
print(f'63.1 Find median of two arrays: index: {input_array1} and {input_array2} in log time:',
      Solution().find_median_index_solution(input_array1, input_array2))
print(f'63.2 Find median of two arrays: {input_array3} and {input_array4} index solution:',
      Solution().find_median(input_array3, input_array4))


""" Binary Search Review """
def binary_search(array, low, high, key):
    while low <= high:
        mid = low + (high - low) // 2

        if key == array[mid]:
            return mid
        elif key > array[mid]:
            low = mid + 1
        else:
            high = mid - 1

    return -1

arr = [1, 4, 5, 7, 10, 12, 14, 17]
target = 5
print(f'binary_search test on {arr} and {target}:', binary_search(arr,0, len(arr)-1, target))


"""
Given an array nums, write a function to move all 0's to the end of it while
maintaining the relative order of the non-zero elements.

Example:

Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
Note:

You must do this in-place without making a copy of the array.
Minimize the total number of operations.
"""


class Solution:
    def move_zeros(self, input_arr):
        current = 0
        last_non_zero = 0

        length = len(input_arr)

        while current < length:
            if input_arr[current] != 0:
                input_arr[last_non_zero] = input_arr[current]
                last_non_zero += 1

            current += 1

        for i in range(last_non_zero, length):
            input_arr[i] = 0

        return input_arr


input_arr = [0, 1, 0, 3, 12]

print(f'64. Move all 0s in array {input_arr} to the end:', Solution().move_zeros(input_arr))

"""
Given an array nums and a value val, remove all instances of that value in-place and return the new length.
Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example 1:
Given nums = [3,2,2,3], val = 3,
Your function should return length = 2, with the first two elements of nums being 2.
It doesn't matter what you leave beyond the returned length.
"""


class Solution:
    def remove_all_instances(self, nums, val):
        left = 0
        right = len(nums) - 1
        length = len(nums)

        while nums[left] != val:
            left += 1
        while nums[right] == val:
            right -= 1
            length -= 1

        while left <= right:
            if nums[left] == val:
                nums[left] = nums[right]
                length -= 1
                right -= 1
            else:
                left += 1

        return length, nums

nums = [3, 2, 2, 3]
val = 3

nums2 = [0, 1, 2, 2, 3, 0, 4, 2]
val2 = 2

print(f'65 Remove all instances of {val} from array {nums} and return new length: ', Solution().remove_all_instances(nums, val))
print(f'65.2 Remove all instances of {val2} from array {nums2} and return new length: ', Solution().remove_all_instances(nums2, val2))


"""
Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""


class Solution:
    def find_k_most_frequent_elements(self, nums, k):
        counter = collections.Counter(nums)
        h = list(counter.items())
        # create min heap
        heapq.heapify(h)

        # logK
        for i in range(len(h) - k):
            heapq.heappop(h)

        res = []
        while len(h):
            res.append(heapq.heappop(h))

        return [it[1] for it in res[::-1]]

    def find_k_most_frequent_elements_max(self, nums, k):
        counter = collections.Counter(nums)
        h = list(counter.items())
        # O(n)
        heapq._heapify_max(h)

        res = []
        # O(klog(n))
        for i in range(k):
            res.append(heapq._heappop_max(h)[1])

        return res


nums = [1, 1, 1, 2, 2, 3]
k = 2

print(f'66 Find K {k} most frequent elements in {nums}: ', Solution().find_k_most_frequent_elements(nums, k))
print(f'66.2 Find K {k} most frequent elements using heapmax in {nums}: ', Solution().find_k_most_frequent_elements(nums, k))

""" 239 Sliding Window Maximum

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
You can only see the k numbers in the window. Each time the sliding window moves right by one position.
Return max from every sliding window.

Example:

Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7]
"""


class Solution:
    def find_max_sliding_window_2(self, nums, k):
        res = []
        q = []

        for i in range(len(nums)):
            q.append(nums[i])
            if len(q) > k:
                q.remove(nums[i-k])

            max_ind = 0
            for ind in range(1, len(q)):
                if q[ind] > q[max_ind]:
                    q = q[ind:]

            if i >= k - 1:
                res.append(q[0])

        return res

    def find_max_sliding_window_deque(self, nums, k):
        candidates = collections.deque()
        output = []

        for i, num in enumerate(nums):
            while candidates and candidates[-1][0] <= num:
                candidates.pop()
            while candidates and candidates[0][1] < i - k + 1:
                candidates.popleft()
            candidates.append((num, i))
            if i >= k - 1:
                output.append(candidates[0][0])
        return output

    def find_max_sliding_window_heap(self, nums, k):
        """ Use heapq
            let's invert the items to form the min heap
            alternative - use _heapify_max or _heappop_max
        """
        for i in range(len(nums)):
            nums[i] *= -1

        min_heap = [(el[1], el[0]) for el in enumerate(nums[:k])]

        heapify(min_heap)

        max_array = [min_heap[0][0]]

        for i in range(k, len(nums)):
            heappush(min_heap, (nums[i], i))

            max_item = min_heap[0]
            # make sure we only get the items from the window of size k
            while max_item[-1] < i - k + 1:
                max_item = heappop(min_heap)

            max_item = min_heap[0]
            max_array.append(max_item[0])

        return list(map(lambda x: x * -1, max_array))


nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
# Output: [3,3,5,5,6,7]


print(f'67.2 Second solution find sliding window of size {k} in array {nums}: {Solution().find_max_sliding_window_2(nums, k)}')
print(f'67.3 Deque solution find sliding window of size {k} in array {nums}: {Solution().find_max_sliding_window_deque(nums, k)}')
print(f'67.4 Heap solution find sliding window of size {k} in array {nums}: {Solution().find_max_sliding_window_heap(nums, k)}')

"""
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:
Input: dividend = 10, divisor = 3
Output: 3

Example 2:
Input: dividend = 7, divisor = -3
Output: -2
"""

import bisect


class Solution:
    def divide_integers(self, dividend, divisor):
        sign = 1
        if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
            sign = -1

        dividend, divisor = abs(dividend), abs(divisor)
        res = 0

        counter, fab = [1, 1], [divisor, divisor]
        while fab[len(fab) - 1] <= dividend:
            fab.append(fab[len(fab)-1] + fab[len(fab)-2])
            counter.append(counter[len(counter) - 1] + counter[len(counter) - 2])

        while dividend >= divisor:
            index = bisect.bisect_right(fab, dividend)
            res += counter[index-1]
            dividend -= fab[index-1]

        if sign == 1:
            if res <= 2147483647:
                return res
            else:
                return 2147483647
        else:
            if res <= 2147483648:
                return -res
            else:
                return -2147483648


dividend = 13
divisor = 3
print(f'68 Divide two integers: {Solution().divide_integers(dividend, divisor)}')

""" 26. Remove Duplicates from Sorted Array
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

Example 1:

Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
Example 2:

Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length.
"""


class Solution:
    def remove_duplicates(self, nums):
        last_unique = 0
        current = 1

        while current < len(nums):
            if nums[current] == nums[last_unique]:
                nums[last_unique + 1] = nums[current + 1]
            else:
                last_unique += 1

            current += 1

        return last_unique + 1


nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]

print(f'69 Remove duplicates from Sorted Array {nums}: {Solution().remove_duplicates(nums)}')


""" 300. Longest Increasing Subsequence
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?
"""


class Solution:
    def find_longest_increasing_seq(self, nums):
        longest_seq = 1

        for i in range(len(nums)):
            current_seq_len = 1
            for j in range(i, len(nums)-1):
                last_seq_element = nums[j]
                if nums[j+1] > last_seq_element:
                    current_seq_len += 1
                    last_seq_element = nums[j+1]

            longest_seq = max(current_seq_len, longest_seq)

        return longest_seq

    """
    Initially, subsequence = [nums[0]]
    Now, for every element from 1 to N-1
        If curr_element > subsequence[-1], then it is greater than the largest element in the subsequence,
            append to the subsequence and increase max length of the subsequence by 1

        If curr_element < subsequence[-1], then we need to find an index in subsequence such that
            subsequence[index] is the first element which is greater than the curr_element.
        Basically, modified binary search on the subsequence array in O(logN) time, note that the subsequence array is guaranteed to be sorted.
        Once we find subsequence[index], we replace the element at that index with curr_element, the size of the max length subsequence does not change.
    """
    def find_element_to_replace(self, find_in_list, element_to_find):
        low = 0
        high = len(find_in_list) - 1

        while low < high:
            mid = low + (high - low)//2

            if find_in_list[mid] < element_to_find:
                low = mid + 1
            else:
                high = mid

        return high

    def find_longest_increasing_seq2(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        subsequence = [nums[0]]
        max_length = 1

        for i in range(1, len(nums)):
            item = nums[i]
            if item > subsequence[-1]:
                max_length += 1
                subsequence.append(item)
            else:
                index_of_element_to_replace = self.find_element_to_replace(subsequence, item)
                subsequence[index_of_element_to_replace] = item

        return max_length

nums = [10, 9, 2, 5, 3, 7, 101, 18]

print(f'71 Find longest increasing subsequence in {nums}: {Solution().find_longest_increasing_seq(nums)}')
print(f'71.2 Find longest increasing subsequence in {nums}: {Solution().find_longest_increasing_seq2(nums)}')


"""
287. Find the duplicate number
Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive),
prove that at least one duplicate number must exist. Assume that there is only one duplicate number,
find the duplicate one.

Example 1:

Input: [1,3,4,2,2]
Output: 2
Example 2:

Input: [3,1,3,4,2]
Output: 3


Approach #3 Floyd's Tortoise and Hare (Cycle Detection) [Accepted]
Intuition

If we interpret nums such that for each pair of index i and value_i
​the "next" value v_j is at index v_i, we can reduce this problem to cycle detection.

Algorithm

First off, we can easily show that the constraints of the problem imply that a cycle must exist.
Because each number in nums is between 1 and n, it will necessarily point to an index that exists.
Therefore, the list can be traversed infinitely, which implies that there is a cycle. Additionally,
because 0 cannot appear as a value in nums, nums[0] cannot be part of the cycle. Therefore,
traversing the array in this manner from nums[0] is equivalent to traversing a cycling linked list.
"""


class Solution:

    def find_duplicate_number(self, nums):
        tortoise = nums[0]
        hare = nums[0]

        # find intersection point of the two runners
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if hare == tortoise:
                break

        # find the "entrance" to the cycle
        ptr1 = nums[0]
        ptr2 = tortoise

        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]

        return ptr1

nums = [3, 1, 3, 4, 2]
nums = [1, 3, 4, 2, 2]
print(f'72 Find duplicate number in {nums}: {Solution().find_duplicate_number(nums)}')


# Python program to print a given number in words.
ones = ["", "one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight ", "nine ", "ten ", "eleven ", "twelve ",
        "thirteen ", "fourteen ", "fifteen ","sixteen ","seventeen ", "eighteen ", "nineteen "]

twenties = ["","","twenty ","thirty ","forty ", "fifty ","sixty ","seventy ","eighty ","ninety "]

thousands = ["","thousand ","million ", "billion ", "trillion ",
             "quadrillion ", "quintillion ", "sextillion ", "septillion ",
             "octillion ", "nonillion ", "decillion ", "undecillion ", "duodecillion "]


def num999(n):
    singles = n % 10  # singles digit
    tens = ((n % 100) - singles) // 10  # tens digit
    hunds = ((n % 1000) - (tens * 10) - singles) // 100  # hundreds digit

    t = ""
    h = ""
    if hunds != 0 and tens == 0 and singles == 0:
        t = ones[hunds] + "hundred "

    elif hunds != 0:
        t = ones[hunds] + "hundred and "

    if tens <= 1:
        h = ones[n % 100]

    elif tens > 1:
        h = twenties[tens] + ones[singles]

    return t + h


def num2word(num):
    if num == 0: return 'zero'
    n = str(num)
    word = ""
    k = 0

    i = 3
    while True:
        nw = n[-i:]
        n = n[:-i]

        word = num999(int(nw)) + thousands[k] + word

        if n == '':
            break
        k += 1

    return word[:-1]

number = 173847

print(f'73 Convert number {number} to words: {num2word(number)}')

""" 295. Find Median from Data Stream
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value.
So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far.


Example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3)
findMedian() -> 2

"""


class Solution:
    def __init__(self, nums=[]):
        self.nums = nums

        # minheap for upper half of elements
        self.min_hi = []

        # maxheap for lower half of vals  - used as minheap with negative vals
        self.max_lo = []

    # insertion sort style
    # can explore binary search as well
    def add_num(self, num):
        self.nums.append(num)
        j = len(self.nums) - 1 - 1

        while num < self.nums[j] and j >= 0:
            self.nums[j+1] = self.nums[j]
            j -= 1

        self.nums[j+1] = num

    def find_median(self):
        print(self.nums)
        l = len(self.nums)
        if l % 2 == 1:
            return self.nums[l//2]

        return (self.nums[l//2-1] + self.nums[l//2])/2

    def add_num_h(self, num):
        heappush(self.max_lo, -num)
        heappush(self.min_hi, -self.max_lo[0])
        heappop(self.max_lo)

        if len(self.max_lo) < len(self.min_hi):
            heappush(self.max_lo, -self.min_hi[0])
            heappop(self.min_hi)

        print('max', self.max_lo, 'min', self.min_hi)

    def find_median_h(self):
        if len(self.min_hi) < len(self.max_lo):
            return -self.max_lo[0]
        else:
            return (self.min_hi[0] - self.max_lo[0]) / 2

s = Solution()
s.add_num_h(1)
s.add_num_h(2)
print(f'74 Find median from data stream: {s.find_median_h()}')
s.add_num_h(3)
print(f'74.2 Find median from data stream: {s.find_median_h()}')

""" 354. Russian Doll Envelops
You have a number of envelopes with widths and heights given as a pair of integers (w, h).
One envelope can fit into another if and only if both the width and height of one
envelope is greater than the width and height of the other envelope.

What is the maximum number of envelopes can you Russian doll? (put one inside other)

Note:
Rotation is not allowed.

Example:

Input: [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7])."""

""" Approach:
    1. Sort list on first/second dimension
    2. find longest increasing subsequence on second
"""


class Solution:
    def find_index(self, seq, el):
        left, right = 0, len(seq) - 1

        while left < right:
            mid = left + (right - left) // 2
            if seq[mid][1] < el[1]:
                left = mid + 1
            else:
                right = mid

        return right

    def find_max_number_envelops(self, nums):
        sorted_envs = sorted(nums, key=lambda x: (x[0], -x[1]))

        longest_seq = [sorted_envs[0]]
        max_no = 1
        for envelop in sorted_envs[1:]:
            if envelop[1] > longest_seq[-1][1]:
                longest_seq.append(envelop)
                max_no += 1
            else:
                env_index = self.find_index(longest_seq, envelop)
                # The conditional statement ensures that the item at index can fit into index+1 on both dimensions
                if envelop[0] < longest_seq[env_index + 1][0]:
                    longest_seq[env_index] = envelop

        return max_no


nums = [[5, 4], [6, 4], [6, 7], [2, 3]]
print(f'75 Russian Doll Envelops for {nums}: max no is {Solution().find_max_number_envelops(nums)}')


""" 127. Word Ladder
 Given two words (beginWord and endWord), and a dictionary's word list,
 find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.

Example 1:
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Example 2:
Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
"""


class Solution:
    # Intuition: BFS on graph via intermediate states
    """
    Algorithm

        1. Do the pre-processing on the given wordList and find all the possible generic/intermediate states.
        Save these intermediate states in a dictionary with key as the intermediate word and value as the list of words which have the same intermediate word.
        2. Push a tuple containing the beginWord and 1 in a queue.
        The 1 represents the level number of a node. We have to return the level of the endNode as that would represent the shortest sequence/distance from the beginWord.
        3. To prevent cycles, use a visited dictionary.
        While the queue has elements, get the front element of the queue. Let's call this word as current_word.
        4. Find all the generic transformations of the current_word and find out if any of these transformations is also a transformation of other words in the word list. This is achieved by checking the all_combo_dict.
        5. The list of words we get from all_combo_dict are all the words which have a common intermediate state with the current_word. These new set of words will be the adjacent nodes/words to current_word and hence added to the queue.
        6. Hence, for each word in this list of intermediate words, append (word, level + 1) into the queue where level is the level for the current_word.
        7. Eventually if you reach the desired word, its level would represent the shortest transformation sequence length.
        8. Termination condition for standard BFS is finding the end word.

    """
    def get_transformations(self, a_word):
        generic_char = '*'
        transformations = []

        for i in range(len(a_word)):
            perm = a_word[:i] + generic_char + a_word[i+1:]
            transformations.append(perm)

        return transformations

    def ladder_length(self, begin_word, end_word, word_list):
        # 1
        permutations_dict = collections.defaultdict(list)
        visited_words = set()

        for a_word in word_list:
            for a_transformation in self.get_transformations(a_word):
                permutations_dict[a_transformation].append(a_word)

        # Optimization: use collections.deque instead of list
        # list: pop() and insert() are O(n) for list
        # deque: popleft() and append() are both O(1)

        q = [(begin_word, 1)]

        while len(q) > 0:
            current_word, current_level = q.pop()
            if current_word == end_word:
                return current_level

            visited_words.add(current_word)

            list_of_word_variants = set()
            transformations = self.get_transformations(current_word)

            for a_trans in transformations:
                list_of_word_variants |= set(permutations_dict[a_trans])
                permutations_dict[a_trans] = []

            # print(permutations_dict)
            # print(list_of_word_variants)

            [q.insert(0, (word, current_level + 1)) for word in list_of_word_variants if word not in visited_words]

        return 0

word_list = ["hot", "dot", "dog", "lot", "log"]
begin_word = 'hit'
end_word = 'cog'

word_list2 = ["hot", "dot", "dog", "lot", "log", "cog"]
begin_word2 = 'hit'
end_word2 = 'cog'

print()
print(f'76 Word Ladder from "{begin_word}" to "{end_word}" and word list {word_list}:'
      f' {Solution().ladder_length(begin_word, end_word, word_list)}')
print(f'76.2 Word Ladder from "{begin_word2}" to "{end_word2}" and word list {word_list2}: '
      f'{Solution().ladder_length(begin_word2, end_word2, word_list2)}')

""" 111. Minimum depth of Binary Tree
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its minimum depth = 2.
"""


class Solution():
    # recursive
    def min_depth(self, root):
        if not root:
            return 0

        if not root.right and not root.left:
            return 1

        elif not root.right:
            return 1 + self.min_depth(root.left)

        elif not root.left:
            return 1 + self.min_depth(root.right)

        return 1 + min(self.min_depth(root.left), self.min_depth(root.right))

    def min_depth_bfs(self, root):
        if not root:
            return 0

        q = collections.deque([(root, 1)])

        while q:
            current, depth = q.popleft()
            if not (current.left or current.right):
                return depth

            if current.left:
                q.append((current.left, depth+1))

            if current.right:
                q.append((current.right, depth+1))


tree = [3, 9, 20, None, None, 15, 7]

root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print()
print(f'77 Min depth of Binary Tree (Recursive) {tree}: {Solution().min_depth(root)}')
print(f'77.2 Min depth of Binary Tree (BFS) {tree}: {Solution().min_depth_bfs(root)}')

"""200. Number of Islands
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1
Example 2:

Input:
11000
11000
00100
00011

Output: 3

"""

input_grid = [['1', '1', '1', '1', '0'], list('11010'), list('11000'), list('00000')]
input_grid2 = [list('11000'), list('11000'), list('00100'), list('00011')]
from typing import List


class Solution:
    # BFS - deque, DFS - list
    def num_islands(self, input_grid: List[List[str]], ) -> int:
        if len(input_grid) == 0:
            return 0

        num_islands = 0

        for y_index, row in enumerate(input_grid):
            for x_index, entry in enumerate(row):
                if entry == '1':
                    q = collections.deque([(x_index, y_index)]) # BFS
                    # q = [(x_index, y_index)] # DFS
                    self.cross_out_island(q, input_grid)
                    num_islands += 1

        return num_islands

    # BFS - deque (popleft(), append()), DFS = list (pop(), append())
    def cross_out_island(self, q: collections.deque, input_grid: List[List[str]]) -> None:
        while len(q):
            # print(q)
            x, y = q.popleft() # BFS
            # x, y = q.pop() # DFS
            input_grid[y][x] = '0'  # use this like "visited" dictionary
            if x > 0 and input_grid[y][x-1] == '1':    # if we can look left and the point is '1'
                input_grid[y][x-1] = '0'    # cross out that point
                q.append((x-1, y))  # add it to the search
            if x < len(input_grid[y]) - 1 and input_grid[y][x+1] == '1':    # if we can look right and the point is '1'
                input_grid[y][x+1] = '0'
                q.append((x+1, y))
            if y > 0 and input_grid[y-1][x] == '1':    # if we can look up and the point is '1'
                input_grid[y-1][x] = '0'
                q.append((x, y-1))
            if y < len(input_grid) - 1 and input_grid[y+1][x] == '1':    # if we can look right and the point is '1'
                input_grid[y+1][x] = '0'
                q.append((x, y+1))

print()
print(f'78 Number of Islands in the grid {input_grid}: {Solution().num_islands(input_grid)}')
print(f'78.2 Number of Islands in the grid {input_grid2}: {Solution().num_islands(input_grid2)}')

print('*** GRAPHS ***')
""" GRAPHS """
graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

""" Find path between two nodes in a Graph"""
def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path

    if start not in graph:
        return None

    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath

    return None


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]

    if start == end:
        return [path]

    if not start in graph:
        return []

    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

    return paths

def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]

    if start == end:
        return path

    if not start in graph:
        return []

    shortest_path = None
    for node in graph[start]:
        if node not in path:
            new_path = find_shortest_path(graph, node, end, path)
            if shortest_path is None or len(new_path) < len(shortest_path):
                shortest_path = new_path

    return shortest_path


def find_shortest_path_bfs(graph, start, end):
        dist = {start: [start]}

        q = collections.deque(start)
        while len(q):
            current = q.popleft()
            for next_node in graph[current]:
                if next_node not in dist:
                    dist[next_node] = dist[current] + [next_node]
                    q.append(next_node)

        return dist[end]

print(f"Path between A and D in graph {graph}:\n {find_path(graph, 'A', 'D')}")
print(f"All paths between A and D in graph {graph}:\n {find_all_paths(graph, 'A', 'D')}")
print(f"Shortest path between A and D in graph {graph}:\n {find_shortest_path(graph, 'A', 'D')}")
print(f"79. Shortest path between A and D in graph BFS {graph}:\n {find_shortest_path_bfs(graph, 'A', 'D')}")


class Graph:
    def __init__(self):
        self.graph_dict = collections.defaultdict(list)

    def add_edge(self, node, neighbour):
        if neighbour not in self.graph_dict.get(node, []):
            self.graph_dict[node].append(neighbour)

    def show_edges(self):
        for node in self.graph_dict:
            for neighbour in self.graph_dict[node]:
                print(f"({node}, {neighbour})")

    def bfs(self, start):
        visited = set()
        visited.add(start)
        q = collections.deque([start])

        while len(q) > 0:
            current = q.popleft()
            for node in self.graph_dict[current]:
                if node not in visited:
                    visited.add(node)
                    q.append(node)

            print(current, end=" ")

g = Graph()
g.add_edge('1', '2')
g.add_edge('1', '3')
g.add_edge('2', '3')
g.add_edge('2', '1')
g.add_edge('3', '1')
g.add_edge('3', '2')
g.add_edge('3', '4')
g.add_edge('4', '3')
g.show_edges()
g.bfs('1')


""" Gas stations
Xenny's is competing in a race and his car has X litres of fuel. There are N milestones in the competition.
It takes no fuel at all to travel between gas stations, but at the  gas station amount of petrol is drained.

Find the number milestones Xenny crosses before his car gets out of fuel.

Input
The first line of input consists of 2 space-separated integers - N and X.

The second line contains N space-separated integers -

Output
Print a single integer - the number of milestones Xenny crosses.
"""


class Solution:
    def get_gas_stations(self, init_gas: int, stations: List) -> int:
        for i in range(len(stations)):
            init_gas -= stations[i]
            if init_gas <= 0:
                return i + 1
        return len(stations)

init_gas = 20
# this indicates how much petrol is drained
stations = [1, 13, 5, 6, 3, 5, 10, 7, 1, 8, 9, 3, 1, 4, 11, 9]


print(f'81. Gas stations before {init_gas} of petrol is drained: {Solution().get_gas_stations(init_gas, stations)}')
print()


"""
    Set of numbers from a list that add up to a certain number
    break down to one_element, the rest of the list should add up to 'target_element - one_element'
"""

def get_number_of_sets(input_list: List, target: int) -> int:
    def rec(arr, total, i):
        if total == 0:
            return 1
        elif total < 0:
            return 0
        elif i < 0:
            return 0

        if total < arr[i]:
            # just exclude last item from math
            return rec(arr, total, i-1)
        else:
            # combo of not_include last item and include last item
            return rec(arr, total, i-1) + rec(arr, total-arr[i], i-1)

    return rec(input_list, target, len(input_list) - 1)


def get_number_of_sets_memoized(input_list: List, target: int) -> int:
    # only total and i change
    mem = {}

    def rec(arr, total, i, mem):
        if (total, i) in mem:
            return mem[(total, i)]

        if total == 0:
            return 1
        elif total < 0:
            return 0
        elif i < 0:
            return 0

        if total < arr[i]:
            # just exclude last item from math
            res = rec(arr, total, i-1, mem)
        else:
            # combo of not_include last item and include last item
            res = rec(arr, total, i-1, mem) + rec(arr, total-arr[i], i-1, mem)

        mem[(total, i)] = res
        return res

    return rec(input_list, target, len(input_list) - 1, mem)

input_list = [2, 4, 6, 10]
target = 16

print(f'82. Number of sets that add up to {target} from {input_list}: {get_number_of_sets_memoized(input_list, target)}')

"""" Num ways to decode a message
 The simple solution will provide O(n^2) time complexity
 Hence, memoization via results list -> reduces complexity to O(n)
"""

import string
vals_dict = dict(zip([str(num) for num in range(1, 27)], string.ascii_lowercase))

cached_dict = {}
def decode(msg, k, memo):
    if k == 0:
        return 1

    s = len(msg) - k # where the str in question begins
    if msg[s] == '0':
        return 0

    if memo[k] is not None:
        # print('got cache for ', k)
        return memo[k]

    result = decode(msg, k-1, memo)
    if k >= 2 and int(msg[s:s+2]) <= 26:
        result += decode(msg, k-2, memo)

    memo[k] = result
    # print('saved cache for ', k)
    return result

def num_ways(message):
    memo = [None]*(len(message)+1)
    return decode(message, len(message), memo)

msg = '12345'
print(f'83. Number of ways to decode message {msg}: {num_ways(msg)}')

""" Given the staircase with N steps, what's the number of ways to climb it if you can step 1 or 2 steps at a time
    How does the problem change if you take any of the num_steps from X, e.g. X = {1, 3, 5}
"""

def make_step(stairs, current_step, memo):
    if stairs - current_step <= 1:
        return 1

    # if stairs - current_step == 2:
    #     return 2

    if memo[current_step] is not None:
        return memo[current_step]

    result = make_step(stairs, current_step+1, memo) + make_step(stairs, current_step+2, memo)
    memo[current_step] = result

    return result

def num_paths(stairs):
    memo = [None] * (stairs+1)
    return make_step(stairs, 0, memo)


def num_paths_bottom_up(stairs):
    if stairs <= 1:
        return 1

    n = [1]*(stairs+1)

    for i in range(2, stairs+1):
        n[i] = n[i-1] + n[i-2]

    return n[stairs]

def num_paths_variation(stairs, steps):
    if stairs == 0:
        return 1

    nums = [1]*(stairs+1)

    for i in range(1, stairs+1):
        total = 0
        for j in steps:
            if i - j >= 0:
                total += num_paths_variation(i - j, steps)
        nums[i] = total

    return nums[stairs]
    # total = 0
    #
    # for a_step in steps:
    #     if n - a_step >= 0:
    #         total += num_paths_variation(n - a_step, steps)
    #
    # return total

stairs = 3
stairs = 4
stairs = 5

steps = {1, 3, 5}
print(f'84. Number of ways to climb the staircase {stairs}: {num_paths(stairs)}')
print(f'84.2 Number of ways to climb the staircase {stairs}: {num_paths_bottom_up(stairs)}')
print(f'84.2 Number of ways to climb the staircase {stairs} with{steps}: {num_paths_variation(stairs, steps)}')


def one_plus(input_array):
    carry = 1

    for i in range(len(input_array)-1, -1, -1):
        total = input_array[i] + carry
        carry = total == 10 and 1 or 0
        input_array[i] = total % 10

    if carry == 1:
        input_array.insert(0, carry)

    return input_array

input1 = [1, 2, 3, 4]
input2 = [0, 1, 9, 9]
input3 = [9, 9, 9]

print(f'85.1 Add 1 to array {input1} : {one_plus(input1)}')
print(f'85.2 Add 1 to array {input2} : {one_plus(input2)}')
print(f'85.3 Add 1 to array {input3} : {one_plus(input3)}')
print()

""" First Recurring Character """


def first_recurring(input_str: str) -> str:
    char_set = set()

    for a_char in input_str:
        if a_char in char_set:
            return a_char

        char_set.add(a_char)

    return 'all chars unique'


input_str1 = 'abcab'
input_str2 = 'badfkjkladf'
input_str3 = 'abcdefghikjlmnop'

print(f'86 First recurring character in a string {input_str1} : {first_recurring(input_str1)}')
print(f'86.2 First recurring character in a string {input_str2} : {first_recurring(input_str2)}')
print(f'86.3 First recurring character in a string {input_str3} : {first_recurring(input_str3)}')


""" All subsets of a list - recursion and iteration"""

def all_subsets(input_list: List) -> None:
    def find_subsets(nums, i, set_so_far, result):
        # print(i, set_so_far)
        if i == len(nums):
            result.add(tuple(set_so_far))

        else:
            find_subsets(nums, i+1, set_so_far, result)
            set2 = set_so_far.copy()
            set2.add(nums[i])
            find_subsets(nums, i+1, set2, result)

    result = set()
    find_subsets(input_list, 0, set(), result)
    return result

def all_subsets_it(input_list: List) -> None:
    subsets = [set()]
    i = 0

    while i < len(input_list):
        new_subsets = []

        for a_set in subsets:
            new_subsets.append(a_set.copy())
            a_set.add(input_list[i])
            new_subsets.append(a_set)

        subsets = new_subsets
        i += 1

    return subsets

input_list = [1, 2, 3]
print(f'87 All subsets of a set{input_list} recursive: {all_subsets(input_list)}')
print(f'87.2 All subsets of a set{input_list} iterative: {all_subsets_it(input_list)}')

""" Find LCA - Lowest Common Ancestor of a tree"""
root = TreeNode(1)
root.left = TreeNode(3)
root.right = TreeNode(2)
root.left.left = TreeNode(4)
root.left.right = TreeNode(6)
root.left.right.left = TreeNode(5)


def get_lca(root, node1, node2):
    def get_path_recursive(root, target, path_so_far=[]):
        if root is None:
            return False
        if root.val == target or (get_path_recursive(root.left, target, path_so_far)
                                  or get_path_recursive(root.right, target, path_so_far)
                                  ):
            path_so_far.insert(0, root.val)
            return True

        return False

    def get_path_iterative(root, target_val):
        q = [(root, [])]

        while q:
            curr_node, curr_path = q.pop()
            curr_path.append(curr_node.val)

            if curr_node.val == target_val:
                return curr_path

            if curr_node.left:
                q.append((curr_node.left, curr_path[:]))
            if curr_node.right:
                q.append((curr_node.right, curr_path[:]))

        return []

    if node1 == node2:
        return node1

    # path1 = []
    # get_path_recursive(root, node1, path1)
    path1 = get_path_iterative(root, node1)

    # path2 = []
    # get_path_recursive(root, node2, path2)
    path2 = get_path_iterative(root, node2)

    if not path1 or not path2:
        return None

    min_len = min(len(path1), len(path2))
    for i in range(min_len - 1, -1, -1):
        if path1[i] == path2[i]:
            return path1[i]

print()
print(f'88 Lowest common ancestor for 4, 5: {get_lca(root, 4,5)}')
print(f'88 Lowest common ancestor for 3, 5: {get_lca(root, 3,5)}')
print(f'88 Lowest common ancestor for 6, 6: {get_lca(root, 6,6)}')

""" Longest Consecutive Char"""

def longest_consecutive_char(input_str):
    longest = 1
    char = input_str[0]
    current_longest = longest

    for i in range(1, len(input_str)):
        if input_str[i] == input_str[i-1]:
            current_longest += 1
            if current_longest > longest:
                longest = current_longest
                char = input_str[i]
        else:
            current_longest = 1

    return longest, char


input_str = 'aabcddbbbea'
print()
print(f'89 Longest Consecutive Char for {input_str}: {longest_consecutive_char(input_str)}')

""" Hopping towers: Start at index 0 and eventually hop outside of the array
    One can only hop nums[i] or less places from any point in the array
"""


def is_hoppable(towers):
    # this will define the optimal next step, from which we can go furthest afterwards
    def find_next_step(position, towers):
        if position + 1 >= len(towers):
            return position + 1

        next_step = position + 1
        max_dist = towers[position+1] + position + 1

        for step in range(2, towers[position]+1):
            if position+step >= len(towers):
                return position + step

            elif towers[position+step] + position + step > max_dist:
                max_dist = towers[position+step] + position + step
                next_step = position + step

        return next_step

    current = 0
    while True:
        if current >= len(towers):
            return True
        if towers[current] == 0:
            return False
        current = find_next_step(current, towers)


def is_hoppable2(towers):
    def helper(towers, current):
        if current >= len(towers):
            return True
        else:
            i = 1
            while i <= towers[current]:
                if helper(towers, current+i):
                    return True
                i += 1

        return False

    current = 0
    res = False
    dist = towers[0]

    for i in range(1, dist+1):
        res = res or helper(towers, i+1)

    return res

def is_hoppable_graph(towers):
    def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path

        if start not in graph:
            return None

        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath:
                    return newpath
        return None

    def find_shortest_path_bfs(graph, start, end):
        dist = {start: [start]}

        q = collections.deque([start])
        while len(q):
            current = q.popleft()
            for next_node in graph[current]:
                if next_node not in dist:
                    dist[next_node] = dist[current] + [next_node]
                    q.append(next_node)

        return dist.get(end)

    g = Graph()
    edge_point = towers[0]

    for i, el in enumerate(towers):
        for j in range(i+1, i+el+1):
            g.add_edge(i, j)
            if j > edge_point:
                edge_point = j

    if edge_point < len(towers):
        return False

    # return find_shortest_path_bfs(g.graph_dict, 0, edge_point)
    return find_path(g.graph_dict, 0, edge_point)

towers = [4, 2, 0, 0, 5, 0]

print(f'90 Tower hopping for {towers} is {is_hoppable2(towers)}')
print(f'90.1 Tower hopping for {towers} graph solution is {is_hoppable_graph(towers)}')

""" Knapsack Problem
You can only put in up to 10Kg in the sack -> need to maximize the value of items
"""
def maximize_val(weight, val, limit):
    def helper(index, current_limit):

        if current_limit <= 0 or index < 0:
            result = 0
        elif weight[index] > current_limit:
            result = helper(index-1, current_limit)
        else:
            tmp1 = helper(index-1, current_limit)
            tmp2 = val[index] + helper(index-1, current_limit-weight[index])
            result = max(tmp1, tmp2)
        return result

    return helper(len(val)-1, limit)

# dynamic programming - only index*current_limit possible vals for helper - memoize it
def maximize_val2(weight, val, limit):
    def helper(index, current_limit):
        if (index, current_limit) in memo:
            return memo[(index, current_limit)]

        if current_limit <= 0 or index < 0:
            result = 0
        elif weight[index] > current_limit:
            result = helper(index-1, current_limit)
        else:
            tmp1 = helper(index-1, current_limit)
            tmp2 = val[index] + helper(index-1, current_limit-weight[index])
            result = max(tmp1, tmp2)

        memo[(index, current_limit)] = result
        return result

    memo = {}
    return helper(len(val)-1, limit)

weight = [1, 2, 4, 2, 5]
value = [5, 3, 5, 3, 2]

print(f'91 Knapsack: maximize value of items up to 10 kg: {maximize_val2(weight, value, 10)}')

""" Longest common subsequence of two strings: not necessarily contiguous
if they end with the same char - add 1 and add result for str before that
if they end with diff char - take max of (substring1 and string2) and (string1, substring2)
"""
str1 = "BATD"
str2 = "ABACD"

def get_longest_common(str1, str2):
    def helper(sub1, sub2):
        if sub1 == '' or sub2 == '':
            return 0

        if sub1[-1] == sub2[-1]:
            return 1 + helper(sub1[:-1], sub2[:-1])

        else:
            tmp1 = helper(sub1, sub2[:-1])
            tmp2 = helper(sub2, sub1[:-1])
            return max(tmp1, tmp2)

    return helper(str1, str2)

def get_longest_common_dynamic(str1, str2):
    def helper(ind1, ind2):
        if memo.get((ind1, ind2)) is not None:
            return memo.get((ind1, ind2))

        if ind1 < 0 or ind2 < 0:
            return 0

        if str1[ind1] == str2[ind2]:
            return 1 + helper(ind1-1, ind2-1)

        else:
            tmp1 = helper(ind1, ind2-1)
            tmp2 = helper(ind1-1, ind2)
            result = max(tmp1, tmp2)
            memo[(ind1, ind2)] = result
            return result

    memo = {}
    return helper(len(str1) - 1, len(str2) - 1)

print(f'92 Get longest common subsequence of two strings: {str1} and {str2}: {get_longest_common_dynamic(str1, str2)}')


""" 733. Flood Fill Problem
An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).

Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.

To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.

At the end, return the modified image.

Example 1:
Input:
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation:
From the center of the image (with position (sr, sc) = (1, 1)), all pixels connected
by a path of the same color as the starting pixel are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected
to the starting pixel.
Note:

The length of image and image[0] will be in the range [1, 50].
The given starting pixel will satisfy 0 <= sr < image.length and 0 <= sc < image[0].length.
The value of each color in image[i][j] and newColor will be an integer in [0, 65535]."""


class Solution():
    def __init__(self, image):
        self.image = image

    def flood_fill(self, sr, sc, new_color):
        if not self.image:
            raise Exception('need an image to flood fill')

        height = len(self.image)
        width = len(self.image[0])

        visited = set()
        q = collections.deque([(sr, sc)])
        self.image[sr][sc] = new_color

        while len(q):
            x, y = q.popleft()
            visited.add((x, y))
            if x-1 >= 0 and self.image[y][x-1] != 0 and (x-1, y) not in visited:
                self.image[y][x-1] = new_color
                q.append((x-1, y))
            if x+1 < width and self.image[y][x+1] != 0 and (x+1,y) not in visited:
                self.image[y][x+1] = new_color
                q.append((x+1, y))
            if y+1 < height and self.image[y+1][x] != 0 and (x, y+1) not in visited:
                self.image[y+1][x] = new_color
                q.append((x, y+1))
            if y-1 >= 0 and self.image[y-1][x] != 0 and (x, y-1) not in visited:
                self.image[y-1][x] = new_color
                q.append((x, y-1))

        return self.image

    def flood_fill_dfs(self, sr, sc, new_color):
        if not self.image:
            raise Exception('need an image to flood fill')

        height = len(self.image)
        width = len(self.image[0])
        visited = set()

        def dfs(r, c, new_color):
            if self.image[r][c] != 0:
                self.image[r][c] = new_color
                visited.add((r, c))

                if r >= 1 and (r-1, c) not in visited:
                    dfs(r-1, c, new_color)
                if r+1 < height and (r+1, c) not in visited:
                    dfs(r+1, c, new_color)
                if c >= 1 and (r, c-1) not in visited:
                    dfs(r, c-1, new_color)
                if c+1 < width and (r, c+1) not in visited:
                    dfs(r, c+1, new_color)

        dfs(sr, sc, new_color)

        return self.image


image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1
sc = 1
new_color = 2

print(f'93 Flood fill for {image} starting at {sr,sc} with {new_color}: {Solution(image).flood_fill(sr, sc, new_color)}')
print(f'93.2 Flood fill for {image} starting at {sr,sc} with {new_color} DFS : '
      f'{Solution(image).flood_fill_dfs(sr, sc, new_color)}')


""" Reserve a room is a scheduling program to return a room number if available
    -2 for invalid input
    -1 if room not available
    min duration is 5 mins
    max duration is 7 days
    rooms can only be reserved on 5 min marks.
"""
import datetime
import bisect


class CandidateSolution:
    def __init__(self):
        self.conf_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
        }

        self.min_duration = 300 # 5 minutes
        self.max_duration = 604800  # 7 days

    @staticmethod
    def get_minutes(timestamp):
        """
            returns minutes from timestamp
        """
        return int(datetime.datetime.fromtimestamp(timestamp).strftime('%M'))

    def valid_input(self, s, e):
        if s > e:
            return False
        if self.get_minutes(s) % 5 != 0 or self.get_minutes(e) % 5 != 0:
            return False
        if e-s < self.min_duration or e-s > self.max_duration:
            return False

        return True

    def find_available_room(self, s, e):

        for a_room, current_times in self.conf_dict.items():
            if len(current_times) == 0:
                current_times.append((s, e))
                return a_room
            else:
                # start time is greater than last meeting's end time
                if s >= current_times[-1][0]:
                    if s >= current_times[-1][1]:
                        current_times.append((s, e))
                        return a_room

                # end time is earlier than first meeting's start time
                elif s < current_times[0][0]:
                    if e <= current_times[0][0]:
                        current_times.insert(0, (s, e))
                        return a_room

                # general case
                else:
                    pos = bisect.bisect(current_times, s)
                    if current_times[pos-1][1] <= s and current_times[pos][0] >= e:
                        bisect.insort(current_times, (s, e))
                        return a_room

        return -1

    def reserve_room(self, start_epochseconds, end_epochseconds):

        if not self.valid_input(start_epochseconds, end_epochseconds):
            return -2

        avail_room = self.find_available_room(start_epochseconds, end_epochseconds)

        return avail_room

s = 1550397600
e = 1550401200
print(f'94. Reserve the room with ({s,e}): room number {CandidateSolution().reserve_room(s, e)}')


""" Peloton: Sort of find max root to leaf problem with 2D matrix
    TODO: implement a graph solution with depth_first to nodes, although the final nodes are somewhat hard to find
    TODO: check for 0s below and don't queue those
"""


class Solution:
    def __init__(self):
        self.wid = 0
        self.height = 0
        self.max = 0

    def iterative_helper(self, arr, j):

        begin_level = 0
        begin_sum = arr[begin_level][j]
        result = begin_sum

        q = collections.deque([(begin_sum, begin_level, j)])

        while q:
            curr_sum, i, j = q.popleft()

            # bottom row of the matrix
            if i == self.height - 1:
                if curr_sum > result:
                    result = curr_sum

            else:
                # check for elements below, right, and left children, add to queue if applicable
                if i + 1 < self.height:
                    if j - 1 >= 0 and arr[i+1][j-1] != 0:
                        q.append((curr_sum + arr[i+1][j-1], i+1, j-1))

                    if j + 1 < self.wid and arr[i+1][j+1] != 0:
                        q.append((curr_sum + arr[i+1][j+1], i+1, j+1))

                    # assumption the element directly under is guaranteed to be non-zero
                    q.append((curr_sum + arr[i+1][j], i+1, j))

        return result

    def recursive_helper(self, arr, i, j, so_far):
        if i == self.height or arr[i][j] == 0:
            return

        so_far += arr[i][j]
        # Leaf Node
        if i == self.height - 1:
            if so_far > self.max:
                self.max = so_far

        else:
            # directly under
            self.recursive_helper(arr, i+1, j, so_far)
            # right child
            if j+1 < self.wid:
                self.recursive_helper(arr, i+1, j+1, so_far)
            # left child
            if j-1 >= 0:
                self.recursive_helper(arr, i+1, j-1, so_far)

    def find_max_path_2d_array_it(self, input_arr):
        self.height = len(input_arr)
        self.wid = len(input_arr[0])

        # find "root"
        j = 0
        for ind, el in enumerate(input_arr[0]):
            if el != 0:
                j = ind
                break

        result = self.iterative_helper(input_arr, j)
        return result

    def find_max_path_2d_array_rec(self, input_arr):
        self.height = len(input_arr)
        self.wid = len(input_arr[0])

        # find "root"
        j = 0
        for ind, el in enumerate(input_arr[0]):
            if el != 0:
                j = ind
                break

        self.recursive_helper(input_arr, 0, j, so_far=0)
        return self.max

input_arr = [[0, 0, 7, 0, 0],
             [0, 4, 7, 0, 0],
             [0, 9, 2, 1, 0],
             [0, 4, 3, 5, 7],
             [1, 8, 6, 3, 5],
             ]

print(f'95. Max path sum from top to bottom in the matrix: recursive: {input_arr}: '
      f'{Solution().find_max_path_2d_array_it(input_arr)}')
print(f'95.2 Max path sum from top to bottom in the matrix: iterative: {input_arr}: '
      f'{Solution().find_max_path_2d_array_rec(input_arr)}')

"""Alien Dictionary - Google Interview Problem
Given a word_list in sorted order, figure out the alphabet

assume we have a Graph with methods
class G:
    def add_edge(self, c1,c2) -> adds edge
    def remove_edge(self, c1,c2) -> removes edge
    def incoming(self, c) -> returns list of chars
    def outgoing(self, c) -> returns list of chars
    def nodes(self) -> returns list of chars

"""


class G:

    def __init__(self):
        self.graph = collections.defaultdict(list)

    def add_edge(self, u, v):
        if v not in self.graph[u]:
            self.graph[u].append(v)

    def remove_edge(self, c1, c2):
        pass

    def incoming(self, c):
        pass

    def outgoing(self, c):
        pass


class Solution():
    def __init__(self):
        self.graph = G()
    """
        this is video/interview implementation on how to create an alphabet once we have the graph
        TODO: look up topological sort
        Runtime Entire Algo: O(len(word_list)*len(longest_word) + O(Y*Y))
    """
    # O(Y*Y) where Y is number of letters
    def pull_alphabet(self, graph):
        output = []

        s = None
        for n in graph.nodes():
            if len(self.incoming(n)) == 0:
                s = n
                break

        zero_nodes = [s]

        while len(zero_nodes):
            zero_node = zero_nodes.pop()
            output.append(zero_node)

            to_remove = self.outgoing(zero_node)
            for a_node in to_remove:
                graph.remove(zero_node, a_node)
                if len(self.incoming(a_node)) == 0:
                    zero_nodes.append(a_node)

        return output

    # def pull_alphabet_copy(self, graph):
    #     output = []
    #
    #   #get the starting node
        # s = None
        # for n in graph.nodes():
        #     if len(self.incoming(n)) == 0:
        #         s = n
        #         break
        #
        # zero_nodes = [s]
        # while len(zero_nodes):
        #     zero_node = zero_nodes.pop()
        #     output.append(zero_node)
        #
        #     to_remove = zero_node.outgoing()
        #     for node in to_remove:
        #         graph.remove(zero_node, node)
        #         if not len(node.incoming()):
        #             zero_nodes.append(node)
        #
        # return output

    def find_alien_dictionary(self, word_list):
        """
        Assume the following algorithm:
            - call function on the wordlist where first chars are compared of word1, word2
                    - if first_c1 == first_c2 -> new_list1 is composed as [word1[1:], word2[1:]]
                    - if first_c1 != first_c2 -> add_edge(first_c1, first_c2)

            - call the function recursively on new_list1, new_list2, etc.
            - stop when len(word1) is 0
        """

        # self.build_relations(word_list)
        self.build_relations_2(word_list, range(len(word_list)), 0)

        print(self.graph.graph)
        # now self.graph.graph contains the graph of letters
        # the first letter of the alphabet must be the one which is absent from values
        # the last letter is the one which is absent from keys
        # find the longest path from first to last will give us alphabet

        all_letters = set(''.join(word_list))
        last_letter = (all_letters - self.graph.graph.keys()).pop()
        set_vals = set()

        for vals in self.graph.graph.values():
            set_vals |= set(vals)

        first_letter = (all_letters - set_vals).pop()

        # longest_path = self.find_longest_path_graph(self.graph.graph, first_letter, last_letter)
        longest_path = self.top_sort(self.graph.graph, first_letter)
        return longest_path

        #Interview method
        # return self.pull_alphabet(self.graph.graph)

    def top_sort(self, g, s):
        stack = []
        visited = set()

        def top_sort_helper(curr_node):
            visited.add(curr_node)

            for a_node in g[curr_node]:
                if a_node not in visited:
                    top_sort_helper(a_node)

            stack.append(curr_node)

        for some_node in list(g.keys())[:]:
            if some_node not in visited:
                top_sort_helper(curr_node=some_node)

        return stack[::-1]

    def find_longest_path_graph(self, g, s, e, path=[]):
        path = path + [s]

        if s == e:
            return path

        if s not in g:
            return []

        longest_path = None

        for a_node in g[s]:
            if a_node not in path:

                new_path = self.find_longest_path_graph(g, a_node, e, path)
                if longest_path is None or len(new_path) > len(longest_path):
                    longest_path = new_path

        return longest_path

    # O(len(word_list)*O(longest_word))
    def build_relations(self, word_list):
        if len(word_list[0]) == 0:
            return

        new_lists = []  # list of lists
        new_list = True

        for i in range(1, len(word_list)):
            if word_list[i][0] == word_list[i-1][0]:

                if new_list is True:
                    new_lists.append([word_list[i-1][1:]])

                new_lists[-1].append(word_list[i][1:])
                new_list = False

            else:
                self.graph.add_edge(word_list[i-1][0], word_list[i][0])
                new_list = True

        for a_list in new_lists:
            self.build_relations(a_list)

    # O(len(word_list)*O(longest_word))
    def build_relations_2(self, word_list, index_list, char_index):
        if char_index == len(word_list[0]):
            return

        char_relations = []
        index_lists_first_char = collections.defaultdict(list)  # 'char': ['word1', 'word2', etc.]

        for i in index_list:
            curr_char = word_list[i][char_index]
            if curr_char not in char_relations:
                char_relations.append(curr_char)

            index_lists_first_char[curr_char].append(i)

        for i in range(1, len(char_relations)):
                self.graph.add_edge(char_relations[i-1], char_relations[i])

        index_lists = [ind_list for ind_list in index_lists_first_char.values() if len(ind_list) > 1]

        for index_list in index_lists:
            self.build_relations_2(word_list, index_list, char_index+1)


word_list = ['bgg', 'fbq', 'fqf', 'ffq', 'gfg']
"""
{'b': ['f'], 'f': ['g', 'q'], 'q': ['f']})
Correct: {‘b’: ['f', 'q'], 'f': ['g'], 'q': ['f']})
"""

print(f'96 Alien dictionary for words: {word_list} is: {Solution().find_alien_dictionary(word_list)}')


""" 199. Binary Tree Right Side View
Given a binary tree, imagine yourself standing on the right side of it,
return the values of the nodes you can see ordered from top to bottom.

Example:

Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]
Explanation:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

Approach track of the depth - and only append to the output last item of any particular depth
BFS and DFS
"""

def construct_tree(input_tree):

    def helper(input_arr, new_root, n , i):
        if i < n:
            if input_arr[i] is None:
                return None
            new_root = TreeNode(input_arr[i])
            new_root.left = helper(input_arr, new_root.left, n, 2*i+1)
            new_root.right = helper(input_arr, new_root.right, n, 2*i+2)

        return new_root

    input_size = len(input_tree)
    root = helper(input_tree, None, input_size, 0)

    return root

def get_right_side_dfs(input_tree):
    #construct the tree from the array
    root = construct_tree(input_tree)

    output = []
    q = [(root, 0)]

    while q:
        node, level = q.pop()
        if level == len(output):
            output.append(node.val)

        if node.left is not None:
            q.append((node.left, level+1))

        if node.right is not None:
            q.append((node.right, level+1))

    return output

def get_right_side_bfs(input_tree):
    root = construct_tree(input_tree)
    output = []
    q = collections.deque([(root, 0)])

    while q:
        node, level = q.popleft()

        if level == len(output):
            output.append(node.val)

        if node.right is not None:
            q.append((node.right, level+1))

        if node.left is not None:
            q.append((node.left, level+1))

    return output

input_tree = [1, 2, 3, None, 5, None, 4]

print(f'97 Binary tree {input_tree} right side view DFS: {get_right_side_dfs(input_tree)}')
print(f'97.2 Binary tree {input_tree} right side view BFS: {get_right_side_bfs(input_tree)}')


""" 101. Symmetric Tree
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3


But the following [1,2,2,null,3,null,3] is not:

    1
   / \
  2   2
   \   \
   3    3
   """

def is_tree_symmetric(input_tree):
    root = construct_tree(input_tree)
    q = collections.deque([(root, 0)])

    level_out = collections.defaultdict(list)

    while len(q):
        node, level = q.popleft()

        level_out[level].append(node.val)

        if node.left:
            q.append((node.left, level+1))

        if node.right:
            q.append((node.right, level+1))

    for level_vals in level_out.values():
        if level_vals[::-1] != level_vals:
            return False

    return True


def is_tree_symmetric_rec(input_tree):
    def is_symmetric(node1, node2):
        if node1 is None and node2 is None:
            return True

        elif node1 is None or node2 is None:
            return False

        elif (node1.val == node2.val
            and is_symmetric(node1.left, node2.right)
            and is_symmetric(node2.left, node1.right)
            ):
            return True

        return False

    root = construct_tree(input_tree)
    return is_symmetric(root.left, root.right)


input1 = [1, 2, 2, 3, 4, 4, 3]
input2 = [1, 2, 2, None, 3, None, 3]

print(f'98.1 Binary tree {input_tree} symmetric?: {is_tree_symmetric(input1)}')
print(f'98.2 Binary tree {input_tree} symmetric recursive ?: {is_tree_symmetric_rec(input1)}')

"""
117. Populating Next Right Pointers in Each Node

Given a binary tree
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

"""
class TreeNodeP(TreeNode):
    def __init__(self, x):
        super().__init__(x)
        self.next = None

def populate_right_pointers(root):
    q = collections.deque([(root, 0)])
    level_dict = collections.defaultdict(list)

    while q:
        node, level = q.popleft()
        level_dict[level].append(node)

        if node.left:
            q.append((node.left, level+1))

        if node.right:
            q.append((node.right, level+1))

    for level_list in level_dict.values():
        if len(level_list) > 1:
            for i in range(1, len(level_list)):
                level_list[i-1].next = level_list[i]

    return root

def populate_right_pointers_no_extra_space(root):
    last_level = 0
    q = collections.deque([(root, last_level)])

    while q:
        node, level = q.popleft()
        if level > last_level:
            curr_ptr = node
            last_level = level

        elif level > 0:
            curr_ptr.next = node
            curr_ptr = node

        if node.left:
            q.append((node.left, level+1))

        if node.right:
            q.append((node.right, level+1))

    return root

root = TreeNodeP(1)
root.right = TreeNodeP(3)
root.left = TreeNodeP(2)
root.left.left = TreeNodeP(4)
root.left.right = TreeNodeP(5)
root.right.right = TreeNodeP(7)

root2 = TreeNodeP(1)
root2.right = TreeNodeP(3)
root2.left = TreeNodeP(2)
root2.left.left = TreeNodeP(4)
root2.left.right = TreeNodeP(5)
root2.right.right = TreeNodeP(7)

print(f'99 Populate Next Right Ptrs Each Node: {populate_right_pointers(root)}')
print(f'99.1 Populate Next Right Ptrs Each Node Save Space: {populate_right_pointers_no_extra_space(root2)}')

""" 301. Remove Invalid Parentheses
Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).

Example 1:

Input: "()())()"
Output: ["()()()", "(())()"]
Example 2:

Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
Example 3:

Input: ")("
Output: [""]
"""


def remove_invalid_parens(input_str):
    """
    Intuition:
        -traverse through the string and get no. of misplaced parens
        -try out ALL the combos of the string resulting from misplaced parens
        -check validity of those combos - if valid, add to result
    """
    left_p = '('
    right_p = ')'
    result = set()

    def get_parens_to_remove(input_str):
        left = right = 0

        for a_char in input_str:
            if a_char == left_p:
                left += 1
            elif a_char == right_p:
                if left > 0:
                    left -= 1
                else:
                    right += 1

        return left, right, left == 0 and right == 0

    def get_all_permutations(sub, ind, valid_len):
        """
            optimization: instead of passing around substring/copies of strings
            pass removed indices instead - compose string on the fly

            optimization: skip non-paren character- just add it
            optimization: keep track of removed left/right parens and don't go down recursion tree if limit reached

        """
        # print(f'calling on sub: {sub} ind: {ind}')
        if ind < 1:
            return

        if len(sub) < valid_len:
            return

        if len(sub) == valid_len:
            _, __, valid_str = get_parens_to_remove(sub)
            if valid_str is True:
                result.add(sub)
        else:
            # keep the char at ind
            get_all_permutations(sub, ind-1, valid_len)
            # print(sub)
            sub2 = sub[:ind-1]+sub[ind:]    # discard the char at ind-1 - this goes from ind==0 up to the end of str
            # print(sub2)
            get_all_permutations(sub2, ind-1, valid_len)
            # print()


    rem_left, rem_right, valid_str = get_parens_to_remove(input_str)

    if valid_str is True:
        return input_str
    if len(input_str) == rem_left + rem_right:
        return ['']

    get_all_permutations(input_str, len(input_str)-1, valid_len=len(input_str) - rem_right - rem_left)
    return result

input_str = "()())()"
input_str2 = "(a)())()"
# input_str3 = ")("
input_str3 = "())))))"

""" Space complexity - O(n) for recursive stack"""
""" Time complexity - O(2^n) , perhaps also multiplied by n as we check valid/invalid"""

print(f'100 Remove Invalid Parentheses and return all possible answers: {remove_invalid_parens(input_str)}')
print(f'100.2 Remove Invalid Parentheses and return all possible answers: {remove_invalid_parens(input_str2)}')
print(f'100.3 Remove Invalid Parentheses and return all possible answers: {remove_invalid_parens(input_str3)}')

""" Matrix Product:
Question: Given a matrix, find the path from top left to bottom right with the
greatest product by moving only down and right, e.g.
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
1 ‐> 4 ‐> 7 ‐> 8 ‐> 9
2016
[‐1, 2, 3]
[4, 5, ‐6]
[7, 8, 9]
‐1 ‐> 4 ‐> 5 ‐> ‐6 ‐>

"""

input_arr = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
input_arr2 = [
    [-1, 2, 3],
    [4, 5, -6],
    [7, 8, 9],
]


def get_matrix_product_iter(input_arr):
    max_v = 2**31-1
    min_v = -2**31+1

    max_dict = {}  # format: '(i, j): [min_val, max_val]'

    for i in range(0, len(input_arr)):
        for j in range(0, len(input_arr[0])):
            max_val = min_v
            min_val = max_v
            curr_element = input_arr[i][j]

            if i == 0 and j == 0:
                max_dict[(i, j)] = [curr_element, curr_element]
                continue

            # get product from above element
            if i > 0:
                temp_min = min(max_dict[(i-1, j)][0] * curr_element, max_dict[(i-1, j)][1] * curr_element)
                temp_max = max(max_dict[(i-1, j)][0] * curr_element, max_dict[(i-1, j)][1] * curr_element)

                max_val = max(max_val, temp_max)
                min_val = min(min_val, temp_min)

            if j > 0:
                temp_min = min(max_dict[(i, j-1)][0] * curr_element, max_dict[(i, j-1)][1] * curr_element)
                temp_max = max(max_dict[(i, j-1)][0] * curr_element, max_dict[(i, j-1)][1] * curr_element)

                max_val = max(max_val, temp_max)
                min_val = min(min_val, temp_min)

            max_dict[(i, j)] = [min_val, max_val]

    return max_dict[(len(input_arr)-1, len(input_arr[0])-1)][1]


print(f'101.3 Matrix Product of {input_arr}: iterative solution: {get_matrix_product_iter(input_arr)}')

def find_path(graph, s, e):
    if s == e:
        return [s]

    visited = set()
    q = collections.deque([(s, [s])])

    while q:
        node, path = q.popleft()

        if node == e:
            return path

        for a_node in graph[node]:
            if a_node not in visited:
                q.append((a_node, path[:]+a_node))
                visited.add(a_node)

    return []


""" 102 Prime Factors Problem - Duolingo
    Time complexity O(nk * log(nk))
"""


def find_n_prime_factors(primes_set, n):
    heap = list(primes_set)
    heapify(heap)
    curr = 0

    while curr < n:
        curr_fact = heappop(heap)
        for next_fact in [curr_fact*a_prime for a_prime in primes_set]:
            heappush(heap, next_fact)
        curr += 1

    return curr_fact

primes = {3, 5, 7}; n = 9
print(f'102 First{n} Primes Factors of the set {primes}: {find_n_prime_factors(primes, n)}')


""" Tracer Interview Problem
Complete the 'moves' function below.
The function is expected to return an INTEGER.
The function accepts INTEGER_ARRAY a as parameter.
It's supposed to return minimal amount of moves required to move all the even elements in an
INT array before all the odd elements
Follow up: don't mutate the initial array: either mutate copy of original array; or advance pointers
without swapping elements
"""


def moves(int_arr):

    left = 0
    right = len(int_arr)-1
    no_moves = 0

    while left <= right:
        if int_arr[left] % 2 == 0:
            left += 1
            continue
        if int_arr[right] % 2 == 1:
            right -= 1
            continue

        # Remove the line below to avoid mutating original list
        int_arr[left], int_arr[right] = int_arr[right], int_arr[left]
        left += 1
        right -= 1

        no_moves += 1

    return no_moves

input_arr1 = [6, 8, 4, 10, 12]
input_arr2 = [6, 8, 1, 10, 12]
assert(moves(input_arr1)) == 0
# assert(moves(input_arr2)) == 1

print(f'103 Min Moves to Arrange Evens before Odds in INT Array {input_arr2}: {moves(input_arr2)}')

""" SquareSpace
Devise effective stack-like data structure to with most time efficient (time complexity O(1)) operations:
    -push/pop
    -remove any element
Don't worry about the space complexity

Solution: the trick is to have some kind of a hybrid structure which allows
    - O(1) access to all elements - HashMap
    - push/pop stack-like operations [list]
    - with efficient remove [doubly linked list]
"""


class EmptyStackException(Exception):
    pass


class DoubleListNode:
    def __init__(self, val, next_node=None, prev_node=None):
        self.val = val
        self.next = next_node
        self.prev = prev_node


class QuickStack:
    def __init__(self):
        self._dict = {}
        self._head = None

    @property
    def keys(self):
        return self._dict.keys()

    @property
    def as_dict(self):
        return self._dict

    def push(self, el):
        new_el = DoubleListNode(el)
        self._dict[el] = new_el

        if not self._head:
            self._head = new_el
        else:
            temp = self._head
            self._head = new_el
            new_el.next = temp
            temp.prev = new_el

    def pop(self):
        if not self._head:
            raise EmptyStackException

        target_el = self._head.val
        if not self._dict.pop(target_el):
            raise EmptyStackException

        self._head = self._head.next
        if self._head:
            self._head.prev = None

        return target_el

    def remove(self, el):
        # head element
        if self._head and el == self._head.val:
            self.pop()
            return

        target_el = self._dict.pop(el)
        if target_el is None:
            raise KeyError

        target_el.prev.next = target_el.next
        # if it's not the last element
        if target_el.next is not None:
            target_el.next.prev = target_el.prev

        # clean up references for garbage collection
        target_el.next = None
        target_el.prev = None

    def peek(self):
        if self._head is not None:
            return self._head.val
        return None


def run_qstack_tests():
    qstack = QuickStack()
    for i in range(1, 10):
        qstack.push(i)
    assert qstack.peek() == 9

    for key in qstack.keys:
        assert qstack.as_dict[key].val == key

    assert qstack.pop() == 9
    assert qstack.pop() == 8
    assert qstack.peek() == 7
    assert qstack.peek() == 7
    assert qstack.remove(5) is None

    try:
        qstack.remove(5)
    except KeyError:
        assert True
    else:
        assert False

    for _ in range(6):
        qstack.pop()

    try:
        qstack.pop()
    except EmptyStackException:
        assert True
    else:
        assert False

run_qstack_tests()

""" Theodo Test: Pawn moving on the chessboard with only i+1,j or j+1 allowed moves
    Find the max amount of grains eaten going from (0, 0) to (M-1, N-1) on MxN board
"""

theodo2 = [
    [2, 2, 4, 2],
    [0, 3, 0, 1],
    [1, 2, 2, 1],
    [4, 1, 2, 2],
]

theodo3 = [
    [2, 2, 4, 2]]

theodo4 = [
    [2], [2], [4], [2]]

def get_matrix_max_sum(input_arr):
    max_dict = {}  # format: '(i, j): max_val'

    for i in range(0, len(input_arr)):
        for j in range(0, len(input_arr[0])):
            curr_element = input_arr[i][j]
            max_val = 0

            if i == 0 and j == 0:
                max_dict[(i, j)] = curr_element
                continue

            if i > 0:
                max_val = max_dict[(i-1, j)] + curr_element

            if j > 0:
                temp_max = max_dict[(i, j-1)] + curr_element
                max_val = max(max_val, temp_max)

            max_dict[(i, j)] = max_val

    return max_dict[(len(input_arr)-1, len(input_arr[0])-1)]


print(f'104. Max chess board move sum on {theodo2}: {get_matrix_max_sum(theodo2)}')
print(f'104.1 Max chess board move sum on {theodo3}: {get_matrix_max_sum(theodo3)}')
print(f'104.2 Max chess board move sum on {theodo4}: {get_matrix_max_sum(theodo4)}')
