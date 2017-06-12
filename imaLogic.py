import math
import cmath

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  Definitions

f = 1+0j
t = complex(-1/2.0,  math.sqrt(3)/2.0)
u = complex(-1/2.0, -math.sqrt(3)/2.0)

def IMGL(X):   return complex(X.real, abs(X.imag))

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  implementations

def NOT(X):    return X**2 * t
def AND(X, Y): return IMGL(X * Y * t).conjugate() * t
def OR(X,Y):   return NOT(AND(NOT(X), NOT(Y)))

def NAND(X,Y): return NOT(AND(X,Y))
def NOR(X,Y):  return NOT(OR(X,Y))

def XOR(X,Y):  return NOT(IMGL(X * Y * u))


#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  circut

def HalfAdder(A,B):
    S = XOR(A,B)
    C = AND(A,B)
    return (S,C)


#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  tests

def disp(X, tol=1e-3):
    if   X.real < 0.0 and X.imag > 0.0: return True
    elif X.real < 0.0 and X.imag < 0.0: return None
    elif X.real > 0.0:                  return False

def dispU(X, tol=1e-3):
    if   X.real < 0.0 and X.imag > 0.0: return "True"
    elif X.real < 0.0 and X.imag < 0.0: return "Unkw"
    elif X.real > 0.0:                  return "Flse"

def bit(X, tol=1e-3):
    return 1 if disp(X) else 0

def bitU(X, tol=1e-3):
    if   dispU(X)[0] == 'T': return 1
    elif dispU(X)[0] == 'F': return 0
    elif dispU(X)[0] == 'U': return 2
    

def printNotTable():
    print """
                NOT
             True|False 
            +----+-----+
            |{t}|{f}|
            +----+-----+
    """.format( t=disp(NOT(t))
              , f=disp(NOT(f))
              )


def make2x2Table(intitle, infunc):
    return """
                {title}
             True |False 
            +-----+-----+
        True|{tt} |{tf}|
       -----+-----+-----+
       False|{ft} |{ff}|
            +-----+-----+
    """.format( title=intitle
              , tt=disp(infunc(t,t))
              , tf=disp(infunc(t,f))
              , ft=disp(infunc(f,t))
              , ff=disp(infunc(f,f))
              )

def make3x3Table(intitle, infunc):
    return """
                {title}
             True|Flse|Unkw 
            +----+----+----+
        True|{tt}|{tf}|{tu}|
       -----+----+----+----+
       False|{ft}|{ff}|{fu}|
       -----+----+----+----+
       Unkwn|{ut}|{uf}|{uu}|
            +----+----+----+
    """.format( title=intitle
              , tt=dispU(infunc(t,t))
              , tf=dispU(infunc(t,f))
              , ft=dispU(infunc(f,t))
              , ff=dispU(infunc(f,f))
              , tu=dispU(infunc(t,u))
              , fu=dispU(infunc(f,u))
              , uf=dispU(infunc(u,f))
              , ut=dispU(infunc(u,t))
              , uu=dispU(infunc(u,u))
              )

def printHalfAdder():
    print "             1/2Adder"
    print "             A|B||C|S"
    for A,B in [(f,f),(f,t),(t,f),(t,t)]:
        S, C = HalfAdder(A,B)
        print "            +-+-++-+-+"
        print "            |{t}|{f}||{c}|{s}|".format( t=bit(A), f=bit(B), c=bit(C), s=bit(S) )
    print "            +-+-++-+-+"


def printHalfAdderU():
    print "             1/2Adder"
    print "             A|B||C|S"
    for A,B in [(f,f),(f,t),(f,u),(t,f),(t,t),(t,u),(u,f),(u,t),(u,u)]:
        S, C = HalfAdder(A,B)
        print "            +-+-++-+-+"
        print "            |{t}|{f}||{c}|{s}|".format( t=bitU(A), f=bitU(B), c=bitU(C), s=bitU(S) )
    print "            +-+-++-+-+"

if __name__=="__main__":
    printNotTable()
    print make2x2Table("AND", AND)
    print make2x2Table("OR", OR)
    print make2x2Table("NAND", NAND)
    print make2x2Table("NOR", NOR)
    print make2x2Table("XOR", XOR)
    printHalfAdder()
    print make3x3Table("AND", AND)
    print make3x3Table("OR", OR)
    print make3x3Table("NAND", NAND)
    print make3x3Table("NOR", NOR)
    print make3x3Table("XOR", XOR)
    printHalfAdderU()
    
    
