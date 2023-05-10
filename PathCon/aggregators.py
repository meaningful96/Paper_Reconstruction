import torch
import torch.nn as nn
from abc import abstractmethod

class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
#------------------------------------------------------------------------------------------------------#

# class Aggregator(nn.Module):
        # def __init__(self ~~):
            # super(Aggregator, self).__init__()

# 이거는 nn.Module에 있는 __init__() 함수를 상속받겠다는 소리이다.
# super(class, self).메서드 를하면
# class가 nn.Module의 메서드를 call해서 상속받는 다는 것을 의미한다.
# 파이토치로 모델 작성시 반드시 써줘야 한다.
# 어지간한 기능이 다 들어가기 때문      

#------------------------------------------------------------------------------------------------------#   
    
    def forward(self, self_vectors, neighbor_vectors, masks):
        # self_vectors:[batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, b_neighbor, 1]
        entitiy_vectors = torch.mean(neighbor_vectors * masks, dim =-2) # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors, entitiy_vectors)
        return outputs
        
        #----------------------------------------------------------------------------------------------#

        # 원래는 Python에서 a 벡터와 b벡터를 곱할 때 1x3 3x1 이런식으로 크기를 맞춰줘야함
        # 따라서 a = 3x1 행렬, b = 1x3행렬이면 a*b = 3x3 행렬이 됨
        # 하지만 pytorch를 사용하면 a*b는 element-wise product가 됨

        #----------------------------------------------------------------------------------------------#
    @abstractmethod
    def _call(self, self_vectors, entitiy_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass
#------------------------------------------------------------------------------------------------------#

# @abstractmethod는 추상 클래스를 만들 때 사용한다.
# 추상 클래스란 메서드의 목록만 가진 클래스이며 상속받는 클래스에서 메서드 구현을 강제하기 위해 사용한다.
# 예를 들어 Person이라는 클래스를 정의할 때, Person의 메서드로 sleep이나 eat를 만들 수는 있지만,
# 사람마다 구체적인 행동 수칙이 다르기 때문에 일반화 하기가 불가능하다.
# 따라서 Person의 eat, sleep등을 추상메서드로 구현하고, 자식 클래스에서 구체적인 행동 수칙을 구현하는 것이다.

#------------------------------------------------------------------------------------------------------#

# 자식 클래스들 구현 Start

class MeanAggregator(Aggregator): # 괄호 안이 Aggregator이므로 Aggregator 클래스의 자식 클래스임을 알 수 있다.
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)
        ## __init__(~~)의 경우 Aggregator의 __init__(~~)과 마찬가지로 ~~의 객체 정보들까지 모두 가지고 온다는 것
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xaier_uniform_(self.layer.weight)
        
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        
        # Mean 연산 수행
        output = torch.mean(entity_vectors, dim = -2) # [batch_size, -1, iuput_dim]
        if self.self_included:
            output += self_vectors
        output = output.view([-1, self.input_dim]) # [-1, input_dim]
        output = self.layer(output) # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.ourput_dim]) # [batch_size, -1, output_dim]
        return self.act(output)
#------------------------------------------------------------------------------------------------------#

# 1) 조건문
# if self.self_included:
    # output += self_vectors
# 이 경우 조건문을 보면 조건이 수식이 아니다.
# 이런 경우 Boole 로서 True or False만을 판단하고, 만약 참일 경우 output에 해당하는 수식을 실행하게 된다.

# 2) view 함수
# reshape와 같은 역할을 하는 함수이다.
# 지금 output의 size가 원래 [batch_size, -1, input_dim]인 텐서인데 여기에
# output = output.view([-1, input_dim])을 하면, [?, input_dim]의 텐서로 크기가 바뀌는 것이다.
# 예를 들어, X가 [3,3,2]인 텐서일 때, X = X.view([-1, 2])를 하면 ?x2가되고 그 총 데이터 수는 동일 => ?==9가된다.
# -1의 의미는, 뒤에 오는 차원의 수에 따라서 알맞게 조정되도록 해 놓는 것이다.

#------------------------------------------------------------------------------------------------------#        

class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act = lambda x:x, self_included = True):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)
        multiplier = 3 if self_included else 2
        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        #----------------------------------------------------------------------------------------------#

        # Xavier initialization을 한 이유(Random intialization을 안쓰는 이유)
        # 노드 수가 많아지면 많이질수록 초깃값을 설정하는 weight가 더 좁게 퍼지게 된다.
        # 따라서 이를 좀 더 완화해주기 위해 Normalization을 하는 Xavier방식을 쓴다.
        #----------------------------------------------------------------------------------------------#
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        
        output = entity_vectors.view([-1, self.input_dim * 2]) # [-1, input_dim * 2]
        if self.self_included:
            # Concatenation 연산 수행
            self_vectors = self_vectors.view([-1, self.input_dim]) # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim = -1) # [-1, input_dim * 3]
        output = self.layer(output) # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim]) # [batch_size, -1, output_dim]
        return self.act(output)
    
class CrossAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act = lambda x: x, self_included = True):
        super(CrossAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)
        addition = self.input_dim if self.self_included else 0
        self.layer = nn.Linear(self.input_dim * self.input_dim + addition, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
    
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        
        # [batch_size, -1, 1, input_dim]
        entity_vectors_a, entity_vectors_b = torch.chunk(entity_vectors, 2, dim = -2)
        entity_vectors_a = entity_vectors_a.view([-1, self.input_dim, 1])
        entity_vectors_b = entity_vectors_b.view([-1, 1, self.input_dim])
        output = torch.matmul(entity_vectors_a, entity_vectors_b) #[ -1, input_dim, input_dim]
        output = output.view([-1, self.input_dim * self.input_dim]) # [-1, inpu_dim*input_dim]
                
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim]) # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim = -1) # [-1, input_dim * input_dim + input_dim]
            #---------------------------------------------------------------------------------------------#
            
            # 예를 들어,
            # a = [-1,input_dim*input_dim] = [5,8*8] = [5,64]이고
            # b = [-1, input_dim] = [5,8] 일 때
            # c = torch.cat([a,b], dim = -1)이면 b출을 기준으로 concatenation하는 것이다. 따라서
            # 사이즈의 첫 번째 부분은 5로 동일해야하며, 그 축을 기준으로 열을 쌓아가는 것이다.
            # 따라서 [5,64] 는 [5, 64 + 8]이 되어 [5,72]의 크기를 가지게 된다.
            
            #---------------------------------------------------------------------------------------------#
        output = self.layer(output) # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim]) # [batch_size, -1, output_dim]
        
        return self.act(output)
    
    
    