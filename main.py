from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason" #name of the agent reasoning node
ACT= "act" #name of the act node    
LAST = -1


def should_continue(state: MessagesState) -> str: #define a function to check if the last message is a tool call
    if not state["messages"][LAST].tool_calls: #if the last message is not a tool call, return END
        return END
    return ACT #if the last message is a tool call, return ACT

flow = StateGraph(MessagesState) #create a state graph with the messages state  

flow.add_node(AGENT_REASON, run_agent_reasoning) #add the agent reasoning node to the flow
flow.set_entry_point(AGENT_REASON) #set the entry point of the flow to the agent reasoning node
flow.add_node(ACT, tool_node) #add the act node to the flow

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END, #if the last message is not a tool call, return END
    ACT:ACT}) #if the last message is a tool call, return ACT

flow.add_edge(ACT, AGENT_REASON) #add the edge from the act node to the agent reasoning node

app = flow.compile() #compile the flow
app.get_graph().draw_mermaid_png(output_file_path="flow.png") #draw the flow as a mermaid png


if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]}) #invoke the flow with the human message  "What is the temperature in Tokyo? List it and then triple it"
    print(res["messages"][LAST].content) #print the last message in the response