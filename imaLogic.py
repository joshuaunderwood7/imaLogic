from __future__ import division
import math
import cmath

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  Definitions

f = 1+0j
t = complex(-1/2.0,  math.sqrt(3)/2.0)
u = complex(-1/2.0, -math.sqrt(3)/2.0)

def IMGL(X):   return complex(X.real, abs(X.imag))

def NORM(Z):   return (Z - f) / (t - f)
def DENM(Z):   return Z * (t - f) + f

def REFINE(Z, tol=1e-3): 
    """
    complex numbers are double precision floats in Python.
    Because of the irrational nature of imaLogic, REFINE 
    should be called periodically to prevent logic "drift"
    """
    if   abs(Z - t) < tol: return t
    elif abs(Z - f) < tol: return f
    elif abs(Z - u) < tol: return u

def IMGB(X):
    """
    IMGB converts true/false imaginary numbers into real numbers
    B(t) = 1 + 0i
    B(f) = 0 + 0i
    """
    return (-2 * (X - 1)) / complex(3, -math.sqrt(3))

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  implementations

def NOT(X):    return X**2 * t
def AND(X, Y): return NOT(IMGL(X * Y * t))
def OR(X,Y):   return NOT(AND(NOT(X), NOT(Y)))

def NAND(X,Y): return NOT(AND(X,Y))
def NOR(X,Y):  return NOT(OR(X,Y))

def XOR(X,Y):  return NOT(IMGL(X * Y * u))

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  circuit

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

def dispN(X):
    return NORM(REFINE(X))

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


def add8Bit( a7, a6, a5, a4, a3, a2, a1, a0
           , b7, b6, b5, b4, b3, b2, b1, b0
           ):
           return ( REFINE(HalfAdder(a7, b7)[0])
                  , REFINE(OR(HalfAdder(a6,b6)[0], HalfAdder(a5,b5)[1]))
                  , REFINE(OR(HalfAdder(a5,b5)[0], HalfAdder(a4,b4)[1]))
                  , REFINE(OR(HalfAdder(a4,b4)[0], HalfAdder(a3,b3)[1]))
                  , REFINE(OR(HalfAdder(a3,b3)[0], HalfAdder(a2,b2)[1]))
                  , REFINE(OR(HalfAdder(a2,b2)[0], HalfAdder(a1,b1)[1]))
                  , REFINE(OR(HalfAdder(a1,b1)[0], HalfAdder(a0,b0)[1]))
                  , REFINE(HalfAdder(a0,b0)[0])
                  )

def addBits(a_bits, b_bits, big_endian=False):
    if big_endian: a_bits.reverse()
    if big_endian: a_bits.reverse()
    C = f # initialize carry bit
    result = []
    for A,B in zip(a_bits, b_bits):
        S, C_n = HalfAdder(A,B)
        # print "HalfAdder({}, {}) = {} ({})".format(disp(A), disp(B), disp(REFINE(S)), disp(C_n)) 
        # print "{}".format(disp(REFINE(OR(REFINE(S),C))))
        result.append(REFINE(OR(REFINE(S),C)))
        C = REFINE(C_n)
    if not big_endian: result.reverse()
    return result, C

def printHalfAdderU():
    print "             1/2Adder"
    print "             A|B||C|S"
    for A,B in [(f,f),(f,t),(f,u),(t,f),(t,t),(t,u),(u,f),(u,t),(u,u)]:
        S, C = HalfAdder(A,B)
        print "            +-+-++-+-+"
        print "            |{t}|{f}||{c}|{s}|".format( t=bitU(A), f=bitU(B), c=bitU(C), s=bitU(S) )
    print "            +-+-++-+-+"

def stardard_test():
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
    

def binary_convert(bits, big_endian=False):
    if not big_endian: bits.reverse()
    return sum(2**i * NORM(b) for (i,b) in enumerate(bits))
    

if __name__=="__main__":
    print "           R=OR( AND(NOT(A), B), AND(A, B) )"
    print "           M=a^48 * b^32 * t"
    print "             A|B||R|M"
    for A,B in [(f,f),(f,t),(f,u),(t,f),(t,t),(t,u),(u,f),(u,t),(u,u)]:
        R = OR( AND(NOT(A), B), AND(A, B) ) 
        M = A**48 * B**32 * t
        print "            +-+-++-+-+"
        print "            |{t}|{f}||{r}|{m}|".format( t=bitU(A), f=bitU(B), r=bitU(R), m=bitU(M) )
    print "            +-+-++-+-+"

    print
    print binary_convert([f,f,f,f, f,f,f,f])
    print binary_convert([f,f,f,f, f,f,f,t])
    print binary_convert([f,f,f,f, f,f,t,f])
    print binary_convert([f,f,f,f, f,f,t,t])
    print binary_convert([f,f,f,f, f,t,f,f])
    print binary_convert([f,f,f,f, f,t,f,t])
    print binary_convert([f,f,f,f, f,t,t,f])
    print binary_convert([f,f,f,f, f,t,t,t])
    print binary_convert([f,f,f,f, t,f,f,f])
    print binary_convert([f,f,f,f, t,f,f,t])
    print binary_convert([f,f,f,f, t,f,t,f])
    print binary_convert([f,f,f,f, t,f,t,t])
    print binary_convert([f,f,f,f, t,t,f,f])
    print binary_convert([f,f,f,f, t,t,f,t])
    print binary_convert([f,f,f,f, t,t,t,f])
    print binary_convert([f,f,f,f, t,t,t,t])

    print binary_convert([f,f,f,t, f,f,f,f])
    print binary_convert([f,f,t,f, f,f,f,f])
    print binary_convert([f,t,f,f, f,f,f,f])
    print binary_convert([t,f,f,f, f,f,f,f])

    print
    print "2 + 2"
    A = [f,f,f,f, f,f,t,f]
    B = [f,f,f,f, f,f,t,f]
    print "{} + {}".format(binary_convert(A), binary_convert(B))
    S,C = addBits(A,B)
    S = map(REFINE, S)
    print binary_convert(S)

    print
    print "42 + 13"
    A = [f,f,t,f, t,f,t,f]
    B = [f,f,f,f, t,t,f,t]
    print "{} + {}".format(binary_convert(A), binary_convert(B))
    S,C = addBits(A,B)
    S = map(REFINE, S)
    print binary_convert(S)

    print
    print "Pure 42 + 13"
    A = [f,f,t,f, t,f,t,f]
    B = [f,f,f,f, t,t,f,t]
    print "{} + {}".format(binary_convert(A), binary_convert(B))
    S = add8Bit(f,f,t,f, t,f,t,f,  f,f,f,f, t,t,f,t)
    print map(disp, S)
    print binary_convert(list(S))

    print "IMGB(t) = {}".format(IMGB(t))
    print "IMGB(f) = {}".format(IMGB(f))


"""

OR( AND(NOT(A), B), AND(A, B) )

IMGL( IMGL( A**2 * B * t**2) * IMGL( A * B * t) * t ) *
IMGL( IMGL( A**2 * B * t**2) * IMGL( A * B * t) * t ) *
t

IMGL(A * B * t) =

A_r B_r -1/2      + 
A_r B_i sqrt(3)i/2 + 
A_i B_r sqrt(3)i/2 + 
A_i B_i -1/2       
+ 
ABS(
A_r B_r sqrt(3)i/2 + 
A_r B_i -1/2      + 
A_i B_r -1/2      + 
A_i B_i sqrt(3)i/2
)i                 


IMGL( A**2 * B * t**2) = IMGL( A**2 * B * u) =

A_r A_r B_r -1/2       +
A_r A_i B_r -1/2       +
A_r A_i B_i -1/2       +
A_i A_r B_i -1/2       +
A_r A_i B_r -sqrt(3)i/2 +
A_i A_r B_r -sqrt(3)i/2 +
A_r A_r B_i -sqrt(3)i/2 +
A_i A_i B_i -sqrt(3)i/2
+
ABS(
A_i A_r B_r -1/2       +
A_i A_i B_r -1/2       +
A_r A_r B_i -1/2       +
A_i A_i B_i -1/2       +
A_r A_r B_r -sqrt(3)i/2 +
A_i A_i B_r -sqrt(3)i/2 +
A_r A_i B_i -sqrt(3)i/2 +
A_i A_r B_i -sqrt(3)i/2
)i


----------------------------      
IMGL( IMGL( A**2 * B * t**2) * IMGL( A * B * t) * t ) =

-1/2         A_r A_r B_r -1/2       +        A_r B_r -1/2      + 
             A_r A_i B_r -1/2       +        A_r B_i sqrt(3)i/2 + 
             A_r A_i B_i -1/2       +        A_i B_r sqrt(3)i/2 + 
             A_i A_r B_i -1/2       +        A_i B_i -1/2        
             A_r A_i B_r -sqrt(3)i/2 +         
             A_i A_r B_r -sqrt(3)i/2 +         
             A_r A_r B_i -sqrt(3)i/2 +         
             A_i A_i B_i -sqrt(3)i/2         
             
-1/2         A_r A_r B_r -1/2       +         ABS(
             A_r A_i B_r -1/2       +         A_r B_r sqrt(3)i/2 +
             A_r A_i B_i -1/2       +         A_r B_i -1/2      +
             A_i A_r B_i -1/2       +         A_i B_r -1/2      +
             A_r A_i B_r -sqrt(3)i/2 +         A_i B_i sqrt(3)/2
             A_i A_r B_r -sqrt(3)i/2 +         )i                 
             A_r A_r B_i -sqrt(3)i/2 +         
             A_i A_i B_i -sqrt(3)i/2         

sqrt(3)i/2   A_r A_r B_r -1/2       +         ABS(
             A_r A_i B_r -1/2       +         A_r B_r sqrt(3)i/2 +
             A_r A_i B_i -1/2       +         A_r B_i -1/2      +
             A_i A_r B_i -1/2       +         A_i B_r -1/2      +
             A_r A_i B_r -sqrt(3)i/2 +         A_i B_i sqrt(3)/2
             A_i A_r B_r -sqrt(3)i/2 +         )i                 
             A_r A_r B_i -sqrt(3)i/2 +         
             A_i A_i B_i -sqrt(3)i/2         
             
sqrt(3)i/2   ABS(                            A_r B_r -1/2      +
             A_i A_r B_r -1/2       +        A_r B_i sqrt(3)i/2 +
             A_i A_i B_r -1/2       +        A_i B_r sqrt(3)i/2 +
             A_r A_r B_i -1/2       +        A_i B_i -1/2       
             A_i A_i B_i -1/2       +
             A_r A_r B_r -sqrt(3)i/2 +
             A_i A_i B_r -sqrt(3)i/2 +
             A_r A_i B_i -sqrt(3)i/2 +
             A_i A_r B_i -sqrt(3)i/2
             )i
+             
ABS(
sqrt(3)i/2   A_r A_r B_r -1/2       +        A_r B_r -1/2      + 
             A_r A_i B_r -1/2       +        A_r B_i sqrt(3)i/2 + 
             A_r A_i B_i -1/2       +        A_i B_r sqrt(3)i/2 + 
             A_i A_r B_i -1/2       +        A_i B_i -1/2        
             A_r A_i B_r -sqrt(3)i/2 +         
             A_i A_r B_r -sqrt(3)i/2 +         
             A_r A_r B_i -sqrt(3)i/2 +         
             A_i A_i B_i -sqrt(3)i/2         
             
             
sqrt(3)i/2   ABS(                            ABS(
             A_i A_r B_r -1/2       +        A_r B_r sqrt(3)i/2 +
             A_i A_i B_r -1/2       +        A_r B_i -1/2      +
             A_r A_r B_i -1/2       +        A_i B_r -1/2      +
             A_i A_i B_i -1/2       +        A_i B_i sqrt(3)i/2
             A_r A_r B_r -sqrt(3)i/2 +        )i                 
             A_i A_i B_r -sqrt(3)i/2 +
             A_r A_i B_i -sqrt(3)i/2 +
             A_i A_r B_i -sqrt(3)i/2
             )i

-1/2         ABS(                            A_r B_r -1/2      +
             A_i A_r B_r -1/2       +        A_r B_i sqrt(3)i/2 +
             A_i A_i B_r -1/2       +        A_i B_r sqrt(3)i/2 +
             A_r A_r B_i -1/2       +        A_i B_i -1/2       
             A_i A_i B_i -1/2       +
             A_r A_r B_r -sqrt(3)i/2 +
             A_i A_i B_r -sqrt(3)i/2 +
             A_r A_i B_i -sqrt(3)i/2 +
             A_i A_r B_i -sqrt(3)i/2
             )i
             
-1/2         ABS(                            ABS(
             A_i A_r B_r -1/2       +        A_r B_r sqrt(3)i/2 +
             A_i A_i B_r -1/2       +        A_r B_i -1/2      +
             A_r A_r B_i -1/2       +        A_i B_r -1/2      +
             A_i A_i B_i -1/2       +        A_i B_i sqrt(3)i/2
             A_r A_r B_r -sqrt(3)i/2 +        )i                 
             A_i A_i B_r -sqrt(3)i/2 +
             A_r A_i B_i -sqrt(3)i/2 +
             A_i A_r B_i -sqrt(3)i/2
             )i
)

 
 
 

          
"""
