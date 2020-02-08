import string
from collections import defaultdict, deque

""" Some Data Structures and Helper Functions"""


class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def insert_item(self, item):
        item.next = self.head
        self.head = item

        return self.head


class ListNode:
    def __init__(self, data):
        self.val = data
        self.next = None


def print_ll(head):
    output = []
    while head is not None:
        output.append(str(head.val))
        head = head.next

    return ', '.join(output)


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def print_bst_by_levels(root):
    if not root:
        return []

    output = []
    q = [(root, 0)]

    while q:
        # could use collections.deque and popleft() for more efficient implementation
        node, level = q.pop(0)
        if level >= len(output):
            output.append([node.val])
        else:
            output[level].append(node.val)

        if node.left:
            q.append((node.left, level+1))
        if node.right:
            q.append((node.right, level+1))

    return output


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, x, y):
        if y not in self.graph[x]:
            self.graph[x].append(y)


""" POW Function O(log(n)) Binary Solution"""
class Solution:
    def power_func(self, x, power):
        if power == 0:
            return 1

        new_pow = power // 2

        temp = self.power_func(x, new_pow)
        if power % 2 == 1:
            return temp * temp * x

        return temp * temp

assert Solution().power_func(2, 11) == 2**11
print(f'11: Power of 2 to 10 is {Solution().power_func(2, 10)}')


""" Reverse Linked List"""


def reverse_list(head):
    current = head
    previous = None

    while current:
        temp = current.next
        current.next = previous
        previous = current
        current = temp

    return previous


def reverse_list_rec(head):
    if head is None or head.next is None:
        return head

    new_head = reverse_list_rec(head.next)
    head.next.next = head
    head.next = None

    return new_head


def compose_ll_from_list(l=[]):
    ll = LinkedList()
    for an_item in l[::-1]:
        ll.insert_item(ListNode(an_item))

    return ll.head

llist = compose_ll_from_list([1, 2, 3, 4, 5])

print(f'21 Reverse Linked List: Original: {print_ll(llist)}')
rev_head = reverse_list(llist)
print(f'21 Reverse Linked List: {print_ll(rev_head)}')
norm_head = reverse_list_rec(rev_head)
print(f'21 Reverse Back to Original: {print_ll(norm_head)}')


""" Palindrome """
def is_palindrome(input_s):
    left_ptr = 0
    right_ptr = len(input_s) - 1

    while left_ptr <= right_ptr:
        if not input_s[left_ptr].isalpha():
            left_ptr += 1
            continue

        if not input_s[right_ptr].isalpha():
            right_ptr -= 1
            continue

        if input_s[left_ptr].lower() != input_s[right_ptr].lower():
            return False

        left_ptr += 1
        right_ptr -= 1

    return True

input_s = 'race a car'
input_s2 = "A man, a plan, a canal: Panama"
print(f'31 Valid Palindrome - {input_s}:'
      f' {is_palindrome(input_s)}')

print(f'31.2 Valid Palindrome - {input_s2}:'
      f' {is_palindrome(input_s2)}')

""" Root to Leaf in a Binary Tree recursive & Iterative """
root = TreeNode(10)
root.left = TreeNode(-2)
root.right = TreeNode(7)
root.left.right = TreeNode(-4)
tnode = TreeNode(8)
root.left.left = tnode
# node that doesn't belong
tnode2 = TreeNode(15)
root.left.right.right = tnode2
tnode3 = TreeNode(43)

def print_leaf_to_node_rec(root, tnode, res=[]):
    if root is None:
        return False

    if (root.val == tnode.val
        or print_leaf_to_node_rec(root.left, tnode, res)
        or print_leaf_to_node_rec(root.right, tnode, res)
        ):
        res.insert(0, root.val)

    return res

def print_leaf_to_node_it(root, tnode):
    """
        DFS search
    :param root:
    :param tnode:
    :return:
    """
    if root is None:
        return

    q = [(root, [])]

    while q:
        curr_node, curr_path = q.pop()
        curr_path.append(curr_node.val)

        if curr_node.val == tnode.val:
            return curr_path

        if curr_node.left is not None:
            q.append((curr_node.left, curr_path[:]))

        if curr_node.right is not None:
            q.append((curr_node.right, curr_path[:]))

    return []


print(f'41 Leaf to Node in a Binary Tree Recursive : {print_leaf_to_node_rec(root, tnode2)}')
print(f'41 Leaf to Node in a Binary Tree Iterative: {print_leaf_to_node_it(root, tnode)}')


""" Remove Element with Value X from Linked List """
def remove_elements(head, to_delete):
    # in case head of the linked list contains value
    while head and head.val == to_delete:
        head = head.next

    if not head:
        return None

    current = head.next
    tracking = head

    while current:
        if current.val == to_delete:
            tracking.next = tracking.next.next
            current = tracking.next
        else:
            tracking = tracking.next
            current = current.next


    # ALTERNATIVE
    # while current and current.next:
    #     while current.next and current.val == to_delete:
    #         current.val = current.next.val
    #         current.next = current.next.next
    #
    #     current = current.next
    #     last_non_val = last_non_val.next
    #
    # if current.val == to_delete:
    #     last_non_val.next = None

    return head

llist = LinkedList()
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(5))
llist.insert_item(ListNode(4))
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(6))
llist.insert_item(ListNode(3))
llist.insert_item(ListNode(2))
llist.insert_item(ListNode(1))

value = 6
print(f'51: Remove nodes from linked list {print_ll(llist.head)} with value {value}: ',
      print_ll(remove_elements(llist.head, value)))


""" Find target in a pivoted array """
def find_target(input_arr, target):

    def find_pivot(input_arr, l, r):

        if input_arr[l] < input_arr[r]:
            return -1

        while l <= r:
            mid = l + (r - l) // 2
            if input_arr[mid+1] < input_arr[mid]:
                return mid

            elif input_arr[mid] > input_arr[l]:
                l = mid + 1
            else:
                r = mid - 1

    def find_index(arr, l, r, target):
        while l <= r:
            mid = l + (r - l) // 2

            if arr[mid] == target:
                return mid

            elif arr[mid] > target:
                r = mid - 1
            else:
                l = mid + 1

        return -1

    pivot = find_pivot(input_arr, 0, len(input_arr) - 1)

    if target == input_arr[pivot]:
        return pivot
    if target < input_arr[0]:
        return find_index(input_arr, pivot+1, len(input_arr)-1, target)
    else:
        return find_index(input_arr, 0, pivot, target)

input = [3, 4, 5, 6, 7, 0, 1, 2]

print(f'15: Target value 0 in rotated sorted array {find_target(input, 0)}')
print(f'15: Target value 7 in rotated sorted array {find_target(input, 7)}')
print(f'15: Target value 1 in rotated sorted array {find_target(input, 1)}')
print(f'15: Target value 5 in rotated sorted array {find_target(input, 5)}')
print(f'15: Target value 3 in rotated sorted array {find_target(input, 3)}')


def max_path_sum_util(root, res):
    #base case
    if root is None:
        return 0

    if root.left is None and root.right is None:
        return root.val

    left_sub_sum = max_path_sum_util(root.left, res)
    right_sub_sum = max_path_sum_util(root.right, res)

    if root.left is not None and root.right is not None:
        res[0] = max(left_sub_sum + right_sub_sum + root.val, res[0])
        return max(left_sub_sum, right_sub_sum) + root.val

    elif root.left is None:
        return root.val + right_sub_sum
    else:
        return root.val + left_sub_sum


def max_path_sum(root):
    res = [-2**32]
    max_path_sum_util(root, res)
    return res[0]

root = TreeNode(-10)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print(f'42 Max Path Sum Between Any Nodes in Binary Tree: {print_bst_by_levels(root)} : {max_path_sum(root)}')


def build_tree(preorder, inorder, pindex):

    if len(inorder) == 1:
        pindex[0] += 1
        return TreeNode(inorder[0])

    new_val = preorder[pindex[0]]
    new_node = TreeNode(new_val)
    pindex[0] += 1

    ind = inorder.index(new_val)

    new_node.left = build_tree(preorder, inorder[:ind], pindex)
    if ind < len(inorder) - 1:
        new_node.right = build_tree(preorder, inorder[ind+1:], pindex)

    return new_node

preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
print('49: Build tree from pre-order and post-order lists :', print_bst_by_levels(build_tree(preorder, inorder, pindex=[0])))


def remove_duplicates(head):
    if not head or not head.next:
        return head

    # when duplicates appear right away
    if head.val == head.next.val:
        while head.next and head.val == head.next.val:
            head = head.next
        head = head.next

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

print(f'52: Remove duplicates from linked list {print_ll(llist.head)}: ', print_ll(remove_duplicates(llist.head)))
print(f'52: Remove duplicates from linked list {print_ll(llist2.head)}: ', print_ll(remove_duplicates(llist2.head)))


def find_longest_increasing_seq(nums):

    def find_sub_index(arr, target):
        low = 0
        high = len(arr) - 1

        while low < high:
            mid = low + (high-low) // 2
            if arr[mid] < target:
                low = mid+1
            else:
                high = mid

        # print(arr, target, low)
        return low

    subseq = [nums[0]]
    no_elems = 1

    for next_num in nums[1:]:
        if next_num > subseq[-1]:
            subseq.append(next_num)
            no_elems += 1

        else:
            ind = find_sub_index(subseq, next_num)
            subseq[ind] = next_num

    return no_elems, subseq

nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f'71.2 Find longest increasing subsequence in {nums}: {find_longest_increasing_seq(nums)}')


def all_subsets(input_list):

    def helper(arr, i, set_so_far, res):
        if i < 0:
            res.add(tuple(set_so_far))
            return

        new_set = set_so_far.copy()
        # print('new_set', new_set)
        helper(arr, i-1, new_set, res)

        set_so_far.remove(arr[i])
        # print('so far', set_so_far)
        helper(arr, i-1, set_so_far, res)


    result = set()
    helper(input_list, len(input_list)-1, set(input_list), result)

    return result


def all_subsets_iter(input_list):
    new_list = [set()]

    for an_item in input_list:
        some_list = list()

        for curr_set in new_list:

            some_list.append(curr_set.copy())
            curr_set.add(an_item)
            some_list.append(curr_set)

        new_list = some_list

    return new_list

input_list = [1, 2, 3]
print(f'87 All subsets of a set{input_list} recursive: {all_subsets(input_list)}')
print(f'87 All subsets of a set{input_list} iter: {all_subsets_iter(input_list)}')


""" Find LCA - Lowest Common Ancestor of a tree"""
def get_lca(root, node1, node2):
    """
        Approach:
        - get path1 and path2 to the nodes
        - traverse path1 and path2 from start and find last path1[i] == path2[i]
    """

    def get_path(root, target_val):
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

    path1 = get_path(root, node1)
    path2 = get_path(root, node2)

    if not path1 or not path2:
        return None

    min_len = min(len(path1), len(path2))
    for i in range(1, min_len):
        if path1[i] != path2[i]:
            return path1[i-1]

    return path1[min_len-1]

root = TreeNode(1)
root.left = TreeNode(3)
root.right = TreeNode(2)
root.left.left = TreeNode(4)
root.left.right = TreeNode(6)
root.left.right.left = TreeNode(5)

print(f'88 Lowest common ancestor for 4, 5: {get_lca(root, 4,5)}')
print(f'88 Lowest common ancestor for 3, 5: {get_lca(root, 3,5)}')
print(f'88 Lowest common ancestor for 6, 6: {get_lca(root, 6,6)}')


""" Hopping towers: Start at index 0 and eventually hop outside of the array
    One can only hop nums[i] or less places from any point in the array
"""

def is_hoppable(towers):
    """
        :param towers:
        :return: True or False

        Approach:
            - recursive approach - try all the hopping options in line with the following base cases:
            - return True if curr_position >= len(towers)
            - return False if the current value is 0
    """
    def helper(towers, current_position):
        if current_position >= len(towers):
            return True
        elif towers[current_position] == 0:
            return False

        pos_increments = towers[current_position]
        for i in range(1, pos_increments+1):
            res = helper(towers, current_position+i)
            if res:
                return True

        return False

    result = helper(towers, 0)
    return result


def is_hoppable_graph(towers):
    """
    Approach:
        - build a graph, make sure keep track of the outer edge
        - see if you can traverse it from 0 to the edge
        - in more general case, one should keep track of the set of outer edges
            that lie beyond len(towers) - reduce the problem to one edge per starting point within len(towers)
    """
    right_border = 0
    g = Graph()

    for i in range(len(towers)):
        max_range = towers[i]

        for j in range(i + 1, i + max_range + 1):
            g.add_edge(i, j)

            # it's enough to get to the outer edge right outside the tower's len
            if right_border < j <= len(towers):
                right_border = j

    # if right border doesn't exceed the len of tower - return False
    if right_border < len(towers):
        return False

    def find_path(graph, s, e, path=[]):
        path += [s]

        if s == e:
            return path

        if s not in graph:
            return None

        for a_node in graph[s]:
            if a_node not in path:
                a_path = find_path(graph, a_node, e, path[:])
                if a_path:
                    return a_path

    def find_path_bfs(graph, s, e):
        if s == e:
            return [s]

        visited = set()
        q = deque([(s, [s])])

        while len(q):
            next_node, path_so_far = q.popleft()
            if next_node == e:
                return path_so_far

            for a_node in graph[next_node]:
                if a_node not in visited:
                    visited.add(a_node)
                    q.append((a_node, path_so_far[:]+[a_node]))

        return None

    path_from_zero_to_edge = find_path(g.graph, 0, right_border)
    path_from_zero_to_edge = find_path_bfs(g.graph, 0, right_border)

    return path_from_zero_to_edge is not None

towers = [4, 2, 0, 0, 5, 0]

print(f'90 Tower hopping for {towers} is {is_hoppable(towers)}')
print(f'90.1 Tower hopping for {towers} graph solution is {is_hoppable_graph(towers)}')

"""
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation
sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
"""


def ladder_length(begin_word, end_word, word_list):
    letters = string.ascii_lowercase
    # shortest_len = 0

    if begin_word == end_word:
        raise Exception('Begin and end words are the same')

    def populate_graph(graph):
        for a_word in [begin_word] + word_list:
            for i in range(len(a_word)):

                for a_letter in letters:
                    a_mutation = a_word[:i] + a_letter + a_word[i+1:]
                    if a_mutation in word_list and a_mutation != a_word:
                        graph.add_edge(a_word, a_mutation)

    def find_ladder_len(graph, s, e):
        visited = set()

        q = deque([(s, 1)])
        while q:
            curr_word, path_len = q.popleft()
            if curr_word == e:
                return path_len

            for a_word in graph.graph[curr_word]:
                if a_word not in visited:
                    visited.add(a_word)
                    q.append((a_word, path_len+1))

        return 0

    g = Graph()
    populate_graph(g)

    return find_ladder_len(g, begin_word, end_word)


word_list = ["hot", "dot", "dog", "lot", "log"]
begin_word = 'hit'
end_word = 'cog'

word_list2 = ["hot", "dot", "dog", "lot", "log", "cog"]
begin_word2 = 'hit'
end_word2 = 'cog'

print(f'76 Word Ladder from "{begin_word}" to "{end_word}" and word list {word_list}:'
      f' {ladder_length(begin_word, end_word, word_list)}')
print(f'76.2 Word Ladder from "{begin_word2}" to "{end_word2}" and word list {word_list2}: '
      f'{ladder_length(begin_word2, end_word2, word_list2)}')


""" Get all combinations of a string """


def get_string_combos(input_str):
    output = set()
    input_len = len(input_str)

    def helper(a_str, l):
        if l == input_len - 1:
            output.add(''.join(a_str))
            # print('adding', a_str)

        for i in range(l, input_len):
            a_str[i], a_str[l] = a_str[l], a_str[i]
            helper(a_str, l+1)
            a_str[i], a_str[l] = a_str[l], a_str[i]

    helper(list(input_str), 0)

    return output, len(output)

input_string = 'abcd'
input_string2 = 'abc'

print(f'102.Get string combos for {input_string}: {get_string_combos(input_string)}')
print(f'102.2 Get string combos for {input_string2}: {get_string_combos(input_string2)}')
