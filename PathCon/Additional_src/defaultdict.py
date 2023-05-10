from abc import abstractmethod
## from abc import abstractmethod
## 이건 추상 클래스를 호출하기 위해 abc 모듈로부터 abstractmethod라는 패키지를 호출 한 것이다.
## 추상 클래스는 Method의 목록만 가진 클래스이며 상속받는 클래스에서 메서드 구현을 강제하기 위해 사용한다.
## 즉 self를 사용하지 않고 함수를 구현한다.
## 추상 클래스는 현재 클래스에서는 행동(함수, Method, Instance)를 지정하기 애매하거나 너무 다양할 때 사용
## 예를 들어 Person이라면 eat, sleep등이 있을 것이다. 그런데 Person이 직접 밥을 어떻게 먹고, 잠을 어떻게
## 자고하는 것을 정의할 수는 없다. 이건 구체적인 각 인간들마다 다르기 때문이다. 그렇기 때문에 Person의 
## eat, sleep등을 추상메서드로 두고 자식 클래스에서 구현하도록 하는 것이다.

# 추상 클래스 예시
from abc import *

class Person(metaclass=ABCMeta):
    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass

class James(Person):
    def eat(self):
        print("chop chop")

    def sleep(self):
        print("coa coa")

class Dean(Person):
    def eat(self):
        print("yam yam")

    def sleep(self):
        print("zzzz")
    

james = James()
dean = Dean()

james.eat() # chop chop
james.sleep() # coa coa
dean.eat() # yam yam
dean.sleep() # zzzz

#%%
## 이런식으로 Person의 eat, sleep을 James와 Dean이 상속받아 오버라이딩하면 된다. 만약 오버라이딩을 하지 않으면 에러가 발생하게
## 된다.

# class Dean(Person):
#     def eat(self):
#         print("yam yam")
        
## Dean 클래스의 sleep오버라이드 부분을 지우면 TypeError: Can't instantiate abstract class Dean with abstract 
## method sleep 에러가 발생한다.
## 추상 클래스의 추상 메서드를 자식 클래스에서 오버라이드하지 않았다는 의미이다

from abc import *

class Person(metaclass=ABCMeta):
    heart = "두근두근"
    mind = ": love, sad, happy, angry"
    countofheart = 1
    
    def readme(self):
        return Person.heart + Person.mind + str(Person.countofheart)
    
    @abstractmethod
    def eat(self):
        pass
    
    @abstractmethod
    def sleep(self):
        pass
    
class James(Person):
    def eat(self):
        print(Person.heart, Person.mind, Person.countofheart)
        print("chop chop")
            
    def sleep(self):
        print(self.readme())
        print("coa coa")
            

james = James()
james.eat()
james.sleep()
