import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from collections import defaultdict

# function 1)
def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count() # cpu 코어 수 확인
    pool = mp.Pool(n_cores)  # 코어 수 만큼 병렬로 프로세싱하기 위해 Pool
    avg = n_triples // n_cores # 코어 당 트리플의 개수를 나눠줌
    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list
#------------------------------------------------------------------------------------------------------------#

# 1) Multiprocessing
    # 대용량 데이터를 효율적으로 처리하기 위해서는 병렬 처리를 활용하는 것이 좋다.
    # Pytorch같은 Framework는 함수 내부에서 병렬 처리를 지원
    # 하지만 데이터 가공 모듈인 numpy나 pandas같은 경우는 별도의 병렬처리가 가능하도록 코딩해야 한다.
    # Pool은 병렬 연산을 지원하는 함수이다.
    # def f(x):
        # return x*x

    # if __name__ == '__main__':
    #   with Pool(5) as p:
            # print(p.map(f, [1, 2, 3]))
            # 출력: [1,4,9]
    
#------------------------------------------------------------------------------------------------------------#

# function 2)
def count_all_paths_with_mp(e2re, max_path_len, head2tails):
    n_cores, pool, range_list = get_params_for_mp(len(head2tails)) # get_params_for_mp가 함수명
    results = pool.map(count_all_paths, zip([e2re] * n_cores,
                                            [max_path_len] * n_cores,
                                            [head2tails[i[0]:i[1]] for i in range_list],
                                            range(n_cores)))  
    #--------------------------------------------------------------------------------------------------------#
    # 1) map함수
    # map(function, iterable)
    # fucntion은 iterable의 각 요소에 적용될 함수이며, iterable은 map() 함수가 적용될 iterable데이터 타입이다.
    # ex)
    #   numbers = [1,2,3,4,5]
    #   doubled = list(map(lambda x:x*2, numbers))
    #   print(doubled) # [2,4,6,8,10]
    
    # 2) zip함수
    # zip함수는 iterable한 데이터 여러 개를 병렬로 iteration해서 하나의 데이터로 엮어주는 것이다.
    # ex)
    #   for item in zip([1,2,3], ['sugar', 'spice', 'everything nice']):
    #       print(item)
    #       # (1, 'sugar')
    #       # (2, 'spice')
    #       # (3, 'everything nice')
    #--------------------------------------------------------------------------------------------------------#
    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)
        
    return res
    #--------------------------------------------------------------------------------------------------------#
    # 1) defaultdict함수
    # collections 모듈에 내장된 함수
    # 파이썬의 내장 자료구조인 사전(dictionary)를 사용하다 보면 어떤 키(key)에 대한 값(value)이 없는 경우가 있음
    # 이런 경우를 처리해 해줘야하는 경우가 따라서 종종 발샘
    # defualtdict는 value가 지정하지 않은 key의 value를 모두 0으로 가져가며 모든 key를 일일이 확인할 필요 x
    #--------------------------------------------------------------------------------------------------------#
    
    
# function 3)
def count_all_paths(inputs):
    # e2re = entitiy2relation
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {} # dictionary 자료 구조형 -> Key, Value
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
        print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    print('pid %d  done' % pid)
    return ht2paths
    

## Breadth-First Search: 너비 우선 탐색
# function 4)
def bfs(head, tails, e2re, max_path_len):
    # put length-1 paths into all_paths
    # each elemnet in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]
    p = 0
    
    # Path의 길이는 Hop수가 2부터 시작
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            entities_in_path = set([head] + [i[1] for i in path])
            for edge in e2re[last_entity_in_path]:
                
                # append (relation, entity) to the path if the new entity does not appear in this path before
                # 이 경로에 이전에 새로운 엔티티가 나타나지 않았다면 (relation, entity)를 추가
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
                
            p += 1
        
    ht2paths = defaultdict(set) # set으로 중복 제거
    for path in all_paths:
        tail = path[-1][1]
        if tail in tails: # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path]))
        
    return ht2paths

#------------------------------------------------------------------------------------------------------------#

# Set함수
# set()은 집합에 관련된 것을 쉽게 처리하기 위해 만든 자료형이다.
# ex)
#   s1 = set([1,2,3])
#   print(s1) # {1,2,3}
# 중복을 허용하지 않는다,
# 순서가 없다.
# 인덱싱으로 접근하려면 리스트나 튜플로 변환한 후 해야 한다.
#------------------------------------------------------------------------------------------------------------#

# function 5)
def count_paths(triplets, ht2paths, train_set):
    res = []
    
    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)]
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])}) 
            #------------------------------------------------------------------------------------------------#
            # set 자료형 연산
            # 겹치는 component만 제거해서 출력
            # ex)
            #   a = {1,2,3}, b = {4,5,6}, c = {1,2,5}
            #   a - b = {1,2,3}, b - a = {4,5,6}
            #   a - c = {3}, c - a = {5}
            #   b - c = {4,6}, c - b = {1,2}
            #------------------------------------------------------------------------------------------------#
        res.append([list(i) for i in path_list])             
        
    return res



# function 6)
def get_path_dict_and_length(train_paths, vaild_paths, test_paths, null_relation, max_path_len):
    # path dictionary와 path length를 얻는다.
    
    path2id = {} # dictionary형태. 참고로, a = {} 이런식으로 비어있는 중괄호 형태면 dictionary로 인식
    id2path = []
    id2length = []
    n_paths = 0
    
    for paths_of_triplet in train_paths + vaild_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path) # path가 dic형태나, set이면 인덱싱을 사용하기 위해선 
                                     # list나 tuple로 바꿔줘야 함.
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                id2path.append(path + [null_relation] * (max_path_len - len(path))) # padding
                n_paths += 1
    return path2id, id2path, id2length
    
    
# function 7)
def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype = np.float64)
    #--------------------------------------------------------------------------------------------------------#
    # 1) sp.lil()
    # 행렬의 값이 대부분 '0' 행렬을 희소행렬(Sparse matrix)라고 한다.
    # 행렬의 값이 대부분 '0'이 아닌 값을 가지는 경우 밀집행렬(Dense matrix)라고 한다.
    # LIL은 row-based format으로 non-zero 요소들을 리스타 튜플과 연결해 저장하는 행렬이다.
    # 각각의 튜플은 두 가지 value값을 포함한다.
    #   - column index
    #   - corresponding value of the non-zeros element
    # LIL형식은 새로운 element를 효율적으로 삽입할 수 있기 때문에 matrix가 점진적(incrementally)으로
    # 구성될 때 유용하다.
    # ex) 
    #   import scipy.sparse as sp
    #   matrix = s.lil_matrix((3,3)) # 3x3행렬 생성, 이 상태로 출력하면 아무것도 출력이 안됨.
    #
    #   matrix[0 ,1] = 2   # 0행 1열에 2라는 요소 삽입
    #   matrix[1, 2] = 3
    #   matrix[2, 0] = 4
    #
    #   print(matrix)
    #   # [[0. 2. 0.]
    #      [0. 0. 3.]
    #      [4. 0. 0.]]
    #--------------------------------------------------------------------------------------------------------#
   
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] =+ 1.0
    return features

# function 8)
def one_hot_paht_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths, path_dict):
        bop_list = [] # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)
        
    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]


# function 9)
def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths ]
            sampled_path_ids_for_triplest = np.random.choice(
                path_ids_for_triplet, size = path_samples, replace = len(path_ids_for_triplet) < path_samples)
        
        path_ids_for_data = np.array(path_ids_for_data, dtype = np.int32)
        res.append(path_ids_for_data)
    return res

# function 10)
def sparse_to_tuple(sparse_matrix):
    if not sp.issmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
        values = sparse_matrix.data
        shape = sparse_matrix.shape
    return indices, values, shape
#------------------------------------------------------------------------------------------------------------#

# sp.issmatrix_coo()
# name.tocoo()
# sp.issmatrix 함수는 행렬을 input으로 받아 input행렬이 sparse matrix인지 아닌지를 Boolean 값으로 리턴
# 만약 Sparse matrix가 맞다면 True, 아니면 False를 리턴
# ex)
#   import numpy as np
#   from scipy import sparse
#
#   # create a sparse matrix using the COO format
#   data = np.array([1, 2, 3])
#   row = np.array([0, 1, 2])
#   col = np.array([0, 1, 2])
#   sparse_matrix = sparse.coo_matrix((data, (row, col)))
#
#   # check if the matrix is sparse
#   is_sparse = sparse.issparse(sparse_matrix)
#   print(is_sparse)  # Output: True

#   # create a dense matrix
#   dense_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#   # check if the matrix is sparse
#   is_sparse = sparse.issparse(dense_matrix)
#   print(is_sparse)  # Output: False

#------------------------------------------------------------------------------------------------------------#
        