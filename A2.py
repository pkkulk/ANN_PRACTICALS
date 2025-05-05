inputs=[[0,0],
        [0,1],
        [1,0],
        [1,1],]
w=[1,-1]
t=1
def MCP(x,w,t):
    s=x[0]*w[0]+x[1]*w[1]
    if s >= t :
        return 1
    else :
        return 0
      

print ("x y output")
for x in inputs:
    mcp = MCP(x,w,t)
    print(x[0] , x[1] , mcp)