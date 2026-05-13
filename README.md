# LangGraph_ExerciseSearcherChat

본 구문은 AI 챗봇 연습을 위해, LangGraph 실습으로 체력 저하 및 체중 증가의 증상을 통해 이를 개선시킬 운동을 찾는 목적으로 실행하였다.

본 AI봇은 VScode 환경에서 conda - Chatbot 가상환경 생성 후, 해당 환경에서 생성된 LLM 모델이다.

### 시기 : 2026-05-13

## Technique Stack :

- langchain_community.llms (Ollama) Ollama LLM 사용
- langchain_core.prompts (PromptTemplate) # langchain의 프롬프트 템플릿
- langgraph.graph (StateGraph, END) # LangGraph 상태 머신
- typing (TypedDict) # 타입 정의

## AI chatbot 모델링

### LangGraph 개요

<img width="108" height="432" alt="image" src="https://github.com/user-attachments/assets/6a7517bb-3392-4031-9fe2-44227e7905b7" />

LLM모델에서 에이전트 내부에서여러 체인과 모듈이 복잡하게 상호작용을 하며, LangGraph는 이를 시각적으로 표현하는 역할을 한다.

LangChain의 핵심 모듈로 개발되어 체인 구성, 디버깅, 실행 등 모든 과정에서 LangChain 기능과 원활하게 연동된다.


### 상태 정의 및 LLM 초기화

LLM을 사용하기 위해서는 사용자 질의, 추출된 내용(본 모델에서는 증상), 검색 결과(본 모델에서는 해결을 위한 운동), 최종응답이 모두 문자형(str)이어야 한다.

따라서 이를 정의하기 위해서 타입 정의를 위한 AgentState 객체를 생성하여, 이를 이용하여 챗봇을 생성한다.

```python
# 1. 상태 정의
class AgentState(TypedDict):  # 상태 타입 정의
    query: str  # 사용자 질의
    symptoms: str  # 추출된 증상
    exercise_candidates: str  # 해결 운동 후보
    result: str  # 최종 응답

# 2. LLM 초기화
llm = Ollama(model="exaone3.5:2.4b")  # Ollama 모델 로딩
```

LLM모델로써, Ollama 모델을 로딩한다. exaone3.5:2.4b 모델을 사용하였다.

### 에이전트 정의

현재 해당 chatbot의 프로세스는 다음과 같다.

1. 질문 입력
2. 증상 추출
3. 증상 해결을 위한 운동 리스트 추출
4. 증상과 운동리스트를 출력

질문 입력은 사용자의 쿼리에 맞추며, 우선은 들어온 질문에 대해서 그 증상을 추출하는 프롬프트를 작성한다.

```python
# 사용자에서 증상을 추출하라고 추출에이전트에 넣을 질의
extractor_prompt = PromptTemplate.from_template("""
                                                사용자의 질문에서 증상에 해당하는 단어 또는 구를 추출.  
                                                결과는 쉼표로 구분된 문자열로 출력.  
                                                질문: {query}
                                                """)  # 증상 추출 프롬프트

# 에이전트 모듈함수 (인자에 반드시 Agent State를 입력해야함. (2 in 1 구조에서 예외가 있기는 하다.))
def extractor_agent(state: AgentState):  # 증상 추출 함수
    chain = extractor_prompt | llm  # 프롬프트 체인 (LCEL)
    symptoms = chain.invoke({"query": state["query"]})  # LLM 실행

    # 사용자 : 나는 체력이 안좋고, 살이 계속 찐다.
    # symptoms : "체력 저하, 체중 증가"
    return {**state, "symptoms": symptoms.strip()}  # 상태에 추가, AgentState class의 state인스턴스에 symptoms 변수값을 수정하는 부분
```

템플릿에 원하는 내용을 입력함으로써, 입력받은 질문에서 증상을 어떤 방식으로 추출할 것이며, 그 결과를 어떻게 출력할 것인지 설정한다.

이런 프롬프트 체인을 이용해서 증상을 추출하면, 추출된 내용을 토대로 해결방안을 위한 운동리스트를 산출하는 단계로 전환한다.

```python
# 추출에이전트에서 뽑은 symptoms를 가지고 해결할 수 있는 운동리스트를 추론하는 의사에이전트.
matcher_prompt = PromptTemplate.from_template("""
                                                다음 증상 목록을 바탕으로 가장 해결 가능성 높은 운동 이름 3개를 쉼표로 추정.
                                                증상: {symptoms}
                                                """)  # 질병 후보 추정 프롬프트

# 추출에이전트에서 얻은 질병을 가지고 해결가능한 운동을 추론하는 파이프함수
def matcher_agent(state: AgentState):  # 질병 후보 추정
    chain = matcher_prompt | llm # LCEL
    candidates = chain.invoke({"symptoms": state["symptoms"]}) # state["symptoms"]: AgentState class의 인스턴스(state)에서
                                                               # symptoms 값을 가져와 질의로 입력
    return {**state, "exercise_candidates": candidates.strip()}
```

템플릿에 원하는 내용을 입력하여, 입력받은 증상들을 이용해 어떤 결과를 어떻게 추출할 것인지를 작성한다.

이렇게 운동 리스트까지 추정이 완료되면 최종적으로 출력할 답변을 정리한다.

```python
# 사용자의 증상과 질병 후보를 받아서 최종답변을 생성하는 에이전트
answer_prompt = PromptTemplate.from_template("""
                                            사용자의 증상: {symptoms}

                                            예측된 운동 후보: {exercise_candidates}

                                            위 내용을 바탕으로 사용자에게 알기 쉽게 개조식으로 설명.
                                            """)  # 최종 응답 생성 프롬프트

# 최종답변 함수
def answer_agent(state: AgentState):  # 응답 생성 에이전트
    chain = answer_prompt | llm  # 프롬프트와 LLM을 연결하여 실행 체인 구성
    answer = chain.invoke({
        "symptoms": state["symptoms"], # Agent State의 인스턴스 sgtate의 symptoms 값 가져옴
        "exercise_candidates": state["exercise_candidates"] # Agent State의 인스턴스 sgtate의 exercise_candidates 값 가져옴
    })
    return {**state, "result": answer.strip()}
```

템플릿에서 증상과 예측된 운동 후보를 입력 받아서, 이를 어떤 방식으로 설명할 것인지 등의 출력 방식을 설정한다.

이렇게 상단에서 사용자의 증상, 운동후보를 가져와서, 이를 통해 답변 내용을 설정해서 최종 출력본을 생성한다.

예시로 다음과 같은 질문을 시행한다.

```python
query = "체력이 안좋고, 살이 계속 찐다"
```

해당 질문을 받으면 위의 에이전트 모델은 체력 저하와 체중 증가에 대한 증상을 추출해, 해당 증상을 토대로 운동리스트 및 설명을 생성하여 답변을 출력할 것이다.

예시 답변은 다음과 같다.

```markdown
============================== 최종 응답:
## 체력 저하 & 체중 증가 개선 계획 (개요)

**현재 상황:**

* **문제:** 체력 저하와 체중 증가가 지속되고 있습니다.
* **목표:** 건강한 생활 습관을 통해 체력 향상과 체중 감량을 도모합니다.

**추천 운동 계획:**

| 운동 종류 | 설명 | 장점 | 주의사항 |
|---|---|---|---|
| **유산소 운동** | 걷기, 달리기, 수영 등 지구력 증진에 효과적입니다. | 심장 건강 개선, 체지방 감소 촉진 | 꾸준한 시간 투자 필요, 적절한 강도 조절 중요 |
| **근력 훈련** | 웨이트 트레이닝 등으로 근력 강화 및 신진대사 촉진 효과를 얻습니다. | 근육량 증가로 기초대사량 상승, 근력 향상 | 올바른 자세 유지, 적절한 무게 조절 필수 |
| **HIIT (고강도 인터벌 트레이닝)** | 짧은 고강도 운동과 휴식을 반복하는 방식으로 효율적인 칼로리 소모 유도. | 시간 효율적, 효과적인 체지방 감소 | 체력 수준 고려한 강도 조절 필수, 초보자는 시작 단계부터 조심 |

**핵심 팁:**

* **꾸준함이 핵심:** 일주일에 3~5회, 꾸준히 운동하는 습관을 들이세요.
* **전문가 상담:** 개인의 체력 수준에 맞는 맞춤형 운동 계획을 위해 의사나 트레이너와 상담하는 것이 좋습니다.
* **균형 잡힌 식단:** 운동과 함께 균형 잡힌 식단을 유지하여 건강한 체중 감량을 지원하세요.

**기억하세요!** 개선 속도는 개인에 따라 다를 수 있습니다. 꾸준히 노력하며 긍정적인 변화를 기대하세요!
```

해당 모델의 경우, 프롬프트 설정을 바꾸어, 증상에 따른 예상 질병 출력, 운동 후 기대효과 구문 추가, 혹은 완전히 새로운 방향으로 내용을 변경할 수 있을것이다.

### 주의 사항

해당 모델은 에이전트간의 연결이 필요하다.

앞서 이미지에서 표시되었듯이, 각 에이전트는 시작에서부터 끝부분까지 연결이 되어있어야, 내용을 받고 출력할 수 있다. 따라서 LangGraph 정의에서 다음과 같은 구문이 필요하다.

```python
from langgraph.graph import StateGraph  # LangGraph 구성 요소

graph = StateGraph(AgentState)  # 그래프 정의
graph.add_node("extractor", extractor_agent)  # 노드 추가
graph.add_node("matcher", matcher_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("extractor")  # 시작 노드 설정
graph.add_edge("extractor", "matcher")  # 노드 간 연결 정의
graph.add_edge("matcher", "answer")
graph.add_edge("answer", END)  # 종료 노드 설정

app = graph.compile()  # 그래프 컴파일
```
