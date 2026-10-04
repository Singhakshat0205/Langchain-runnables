from langchain_core.runnables import RunnableLambda 

def word_counter(text):
    return len(text.split())

runnable_word_counter= RunnableLambda(word_counter)

print(runnable_word_counter.invoke('Hi my name is akshat singh I am an AI engineer with 12 lpa package'))
