from fastmcp import FastMCP

mcp = FastMCP('Calculator')

@mcp.tool
def add(num1: int, num2: int):
    """add any two numbers of type int."""
    return num1 + num2
@mcp.tool
def sub(num1: int, num2: int):
    """subtract any two numbers of type int."""
    return num1 - num2
@mcp.tool
def mul(num1: int, num2: int):
    """multiply any two numbers of type int."""
    return num1 * num2

if __name__ == '__main__':
    mcp.run(transport='stdio')