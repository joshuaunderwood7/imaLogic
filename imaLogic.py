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

def RIML(X):   return REFINE(IMGL(X))
def RNRM(X):   return REFINE(NORM(X))
def RDEN(X):   return REFINE(DENM(X))

def IMGB(X):
    """
    IMGB converts true/false imaginary numbers into real numbers
    B(t) = 1 + 0i
    B(f) = 0 + 0i
    """
    return (-2 * (X - 1)) / complex(3, -math.sqrt(3))

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  implementations

def UNK(X):     return X**2 * f
def NOT(X):     return X**2 * t
def IF(X):      return X**2 * u

def OR(X,Y):    return IMGL(X * Y)# * f)
def NOR(X,Y):   return NOT(OR(X,Y))
    
def NAND(X,Y):  return IMGL(X * Y * t)
def AND(X, Y):  return NOT(NAND(X,Y))
                                                                                                                                                                            
def XNOR(X,Y):  return IMGL(X * Y * u)
def XOR(X,Y):   return NOT(XNOR(X,Y))

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  Decimal integer equality
                                                                                                                                                                            
def CMP(A : int, B : int, depth=0):
    """
    returns "true" if A == B

    This function cheats by using an if-statement/piece-wise function. 
    So, don't use this one for actual imaLogic proof
    """                                                                                                                                                                                                                                                                          
    if  A==0 and B==0: return t
    else:
        return REFINE(AND( REFINE(XNOR( DENM(A%2), DENM(B%2) )), CMP(A//2, B//2, depth+1) ))

def CMP16(A : int, B : int):
    Ab = [ DENM( (A // 2**0 ) % 2 )
         , DENM( (A // 2**1 ) % 2 )
         , DENM( (A // 2**2 ) % 2 )
         , DENM( (A // 2**3 ) % 2 )
         , DENM( (A // 2**4 ) % 2 )
         , DENM( (A // 2**5 ) % 2 )
         , DENM( (A // 2**6 ) % 2 )
         , DENM( (A // 2**7 ) % 2 )
         , DENM( (A // 2**8 ) % 2 )
         , DENM( (A // 2**9 ) % 2 )
         , DENM( (A // 2**10) % 2 )
         , DENM( (A // 2**11) % 2 )
         , DENM( (A // 2**12) % 2 )
         , DENM( (A // 2**13) % 2 )
         , DENM( (A // 2**14) % 2 )
         , DENM( (A // 2**15) % 2 )
         ]
    Bb = [ DENM( (B // 2**0 ) % 2 )
         , DENM( (B // 2**1 ) % 2 )
         , DENM( (B // 2**2 ) % 2 )
         , DENM( (B // 2**3 ) % 2 )
         , DENM( (B // 2**4 ) % 2 )
         , DENM( (B // 2**5 ) % 2 )
         , DENM( (B // 2**6 ) % 2 )
         , DENM( (B // 2**7 ) % 2 )
         , DENM( (B // 2**8 ) % 2 )
         , DENM( (B // 2**9 ) % 2 )
         , DENM( (B // 2**10) % 2 )
         , DENM( (B // 2**11) % 2 )
         , DENM( (B // 2**12) % 2 )
         , DENM( (B // 2**13) % 2 )
         , DENM( (B // 2**14) % 2 )
         , DENM( (B // 2**15) % 2 )
         ]
    return ANDR( AND( AND( ANDR( XNOR(Ab[ 0], Bb[ 0]), XNOR(Ab[ 1], Bb[ 1]) )
                         , ANDR( XNOR(Ab[ 2], Bb[ 2]), XNOR(Ab[ 3], Bb[ 3]) ) 
                         )
                    , AND( ANDR( XNOR(Ab[ 4], Bb[ 4]), XNOR(Ab[ 5], Bb[ 5]) )
                         , ANDR( XNOR(Ab[ 6], Bb[ 6]), XNOR(Ab[ 7], Bb[ 7]) ) 
                    )    )
               , AND( AND( ANDR( XNOR(Ab[ 8], Bb[ 8]), XNOR(Ab[ 9], Bb[ 9]) )
                         , ANDR( XNOR(Ab[10], Bb[10]), XNOR(Ab[11], Bb[11]) ) 
                         )
                    , AND( ANDR( XNOR(Ab[12], Bb[12]), XNOR(Ab[13], Bb[13]) )
                         , ANDR( XNOR(Ab[14], Bb[14]), XNOR(Ab[15], Bb[15]) ) 
               )    )    )

def CMP8(A : int, B : int):
    Ab = [ DENM( (A // 2**0) % 2 )
         , DENM( (A // 2**1) % 2 )
         , DENM( (A // 2**2) % 2 )
         , DENM( (A // 2**3) % 2 )
         , DENM( (A // 2**4) % 2 )
         , DENM( (A // 2**5) % 2 )
         , DENM( (A // 2**6) % 2 )
         , DENM( (A // 2**7) % 2 )
         ]
    Bb = [ DENM( (B // 2**0) % 2 )
         , DENM( (B // 2**1) % 2 )
         , DENM( (B // 2**2) % 2 )
         , DENM( (B // 2**3) % 2 )
         , DENM( (B // 2**4) % 2 )
         , DENM( (B // 2**5) % 2 )
         , DENM( (B // 2**6) % 2 )
         , DENM( (B // 2**7) % 2 )
         ]
    return ANDR( AND( ANDR( XNOR(Ab[0], Bb[0]), XNOR(Ab[1], Bb[1]) )
                    , ANDR( XNOR(Ab[2], Bb[2]), XNOR(Ab[3], Bb[3]) ) 
                    )
               , AND( ANDR( XNOR(Ab[4], Bb[4]), XNOR(Ab[5], Bb[5]) )
                    , ANDR( XNOR(Ab[6], Bb[6]), XNOR(Ab[7], Bb[7]) ) 
                    )
               )

def CMP4(A : int, B : int):
    Ab = [ DENM( (A // 2**0) % 2 )
         , DENM( (A // 2**1) % 2 )
         , DENM( (A // 2**2) % 2 )
         , DENM( (A // 2**3) % 2 )
         ]
    Bb = [ DENM( (B // 2**0) % 2 )
         , DENM( (B // 2**1) % 2 )
         , DENM( (B // 2**2) % 2 )
         , DENM( (B // 2**3) % 2 )
         ]
    return ANDR( AND( XNOR(Ab[0], Bb[0]), XNOR(Ab[1], Bb[1]) )
               , AND( XNOR(Ab[2], Bb[2]), XNOR(Ab[3], Bb[3]) ) 
               )

def CMP2(A : int, B : int):
    Ab = [ DENM( (A // 2**0) % 2 )
         , DENM( (A // 2**1) % 2 )
         ]
    Bb = [ DENM( (B // 2**0) % 2 )
         , DENM( (B // 2**1) % 2 )
         ]
    return ANDR( XNOR(Ab[0], Bb[0]), XNOR(Ab[1], Bb[1]) )

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Memory manipulations

def MEM_WRITE(MEM_MAT, ADDR, NEW_VAL, bitwidth=8):
    """
    I think I can get away with the list comprehension here:
    
    This would be the same if written out for each line of the matrix
    and I don't want a 64k-line matrix in my source code
    """
    if   bitwidth== 2: CMPF = CMP2
    elif bitwidth== 4: CMPF = CMP4
    elif bitwidth== 8: CMPF = CMP8
    elif bitwidth==16: CMPF = CMP16
    else             : CMPF = CMP

    return [ [ row[0]
             , ( NEW_VAL * RNRM(    CMPF(ADDR, row[0] )).real 
               + row[1]  * RNRM(NOT(CMPF(ADDR, row[0]))).real 
               ) 
             ]
             for row in MEM_MAT
           ]

def MEM_READ(MEM_MAT, ADDR, bitwidth=8):
    """
    I think I can get away with the list comprehension here:
    
    This would be the same if written out for each line of the matrix
    and I don't want a 64k-line matrix in my source code
    """
    if   bitwidth== 2: CMPF = CMP2
    elif bitwidth== 4: CMPF = CMP4
    elif bitwidth== 8: CMPF = CMP8
    elif bitwidth==16: CMPF = CMP16
    else             : CMPF = CMP

    return sum( row[1] * RNRM(CMPF(ADDR, row[0])).real 
                for row in MEM_MAT
              )

def MEM_REFINE(MEM):
    return [[a,int(v)] for a,v in MEM]
    
#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Brainf*ck python adapters

from functools import partial
def ID(x): 
    return x
    
def BF_MULTIPLY(LAMBDA, X):
    if X == 0:
        return []
    else:
        return LAMBDA() * X
        
class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

getch = _Getch()
    
#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Brainf*ck definition and operations

PR = ord(">")
PL = ord("<")
PI = ord("+")
PD = ord("-")
LB = ord("[")
RB = ord("]")
OP = ord(".")
IP = ord(",")
BF_CMDS     = [PR, PL, PI, PD, LB, RB]
EXTENDED_BF = [PR, PL, PI, PD, LB, RB, OP, IP]

# Assign memory limitation
# While unbounded memory is a feature, practical examples
# would require programs to exist in a specified size of 
# memory.  Here's I'm choosing 8-bits to store the program
# and the output
ADDRESS_WIDTH = 8
MEM_SIZE = 2**ADDRESS_WIDTH

# Define the program
# I'd love to start with hello world, but I found a smaller
# program on esolangs.org that does not require I/O (which 
# is not required to be turing complete).  This program uses
# all six Brainf*ck commands to move a value "two places to 
# the right" in memory
PROGRAM = "[>>[-]<<[->>+<<]]"
PROGRAM_LEN = len(PROGRAM)

# Reserve addresses for Brainf*ck operation:
# The Program pointer -> where are we in the program
PROG_P  = 0
# The stack counter's pointer -> How deep are we nested in "[]" blocks
STACK_P = 1
# The end of the Brainf*ck program.  (start of the BF memory) for convinence
BF_HALT = 2
# The Brainf*ck pointer, allowes program manipulations of memory
BF_P    = 3

BF_MEM = ([ [ 0 , 4 ]  # The program counter (Program starts at address 3)
          , [ 1 , 0 ]  # The stack counter used to count nested contitionals' parens 
          , [ 2 , 4 + PROGRAM_LEN]  # This is the Brainf*ck HALT address
          , [ 3 , 4 + PROGRAM_LEN]  # This is the Brainf*ck "pointer"
          ]
          + [ [ii, ord(c)] for ii,c in enumerate(PROGRAM, 4)] # Load the Program
          + [ [ii, 0] for ii in range(4+PROGRAM_LEN, MEM_SIZE)] # Fill the rest of the memory with zeros
         )
BF_MEM = MEM_WRITE(BF_MEM, MEM_READ(BF_MEM, BF_P), 3) # Put a value in memory (at *BF_P = 3) to "move to the right"

def LOAD_PROGRAM(prg_str):
    global PROGRAM
    global PROGRAM_LEN
    global BF_MEM
    PROGRAM = prg_str
    PROGRAM_LEN = len(PROGRAM)
    BF_MEM = ([ [ 0 , 4 ]  # The program counter (Program starts at address 3)
              , [ 1 , 0 ]  # The stack counter used to count nested contitionals' parens 
              , [ 2 , 4 + PROGRAM_LEN]  # This is the Brainf*ck HALT address
              , [ 3 , 4 + PROGRAM_LEN]  # This is the Brainf*ck "pointer"
              ]
              + [ [ii, ord(c)] for ii,c in enumerate(PROGRAM, 4)] # Load the Program
              + [ [ii, 0] for ii in range(4+PROGRAM_LEN, MEM_SIZE)] # Fill the rest of the memory with zeros
             )

def PTR_RIGHT(MEM):
    return MEM_WRITE(MEM, BF_P, MEM_READ(MEM, BF_P) + 1)

def PTR_LEFT(MEM):
    return MEM_WRITE(MEM, BF_P, MEM_READ(MEM, BF_P) - 1)

def PTR_INC(MEM):
    return MEM_WRITE(MEM, MEM_READ(MEM, BF_P),  MEM_READ(MEM, MEM_READ(MEM, BF_P)) + 1)

def PTR_DEC(MEM):
    return MEM_WRITE(MEM, MEM_READ(MEM, BF_P),  MEM_READ(MEM, MEM_READ(MEM, BF_P)) - 1)

def SCAN_LEFT(MEM):
    MEM = MEM_WRITE( MEM, STACK_P
                   , int( ( MEM_READ(MEM, STACK_P)                                        ) 
                        - ( 1 * RNRM(CMP8(MEM_READ(MEM,  MEM_READ(MEM,PROG_P)), LB)).real ) 
                        + ( 1 * RNRM(CMP8(MEM_READ(MEM,  MEM_READ(MEM,PROG_P)), RB)).real ) 
                        )
                   )
    # Go one past the matching left bracket in anticipation of the PC++
    MEM = MEM_WRITE(MEM, PROG_P,  MEM_READ(MEM, PROG_P) - 1)
    
    # While this is most correct, the python interpreter won't short-circut 
    # a multiply-by-zero, so I added a BF_MULTIPLY function to do just that.
    # A more sophisticated multiplication system or a human could make this 
    # short circut without the if-else block in BF_MULTIPLY
    #return ( MEM * RNRM(CMP8(MEM_READ(MEM,  STACK_P), 0)).real ) + ( SCAN_LEFT(MEM) * RNRM(NOT(CMP8(MEM_READ(MEM,  STACK_P), 0))).real )
    return ( (BF_MULTIPLY(partial(       ID,MEM), int(RNRM(    CMP8(MEM_READ(MEM,  STACK_P), 0) ).real)) )  
           + (BF_MULTIPLY(partial(SCAN_LEFT,MEM), int(RNRM(NOT(CMP8(MEM_READ(MEM,  STACK_P), 0))).real)) )
           )
# I made a choice to always scan on a right bracket, the conditional is only on LEFT_BRACKET
RIGHT_BRACKET = SCAN_LEFT


def SCAN_RIGHT(MEM):
    MEM = MEM_WRITE( MEM, STACK_P
                   , int( ( MEM_READ(MEM, STACK_P)                                        ) 
                        + ( 1 * RNRM(CMP8(MEM_READ(MEM,  MEM_READ(MEM,PROG_P)), LB)).real ) 
                        - ( 1 * RNRM(CMP8(MEM_READ(MEM,  MEM_READ(MEM,PROG_P)), RB)).real ) 
                        )
                   )
    # Stop on the matching right bracket in anticipation of the PC++
    MEM = MEM_WRITE(MEM, PROG_P,  MEM_READ(MEM, PROG_P) + int(RNRM(NOT(CMP8(MEM_READ(MEM,  STACK_P), 0))).real))
    
    # While this is most correct, the python interpreter won't short-circut 
    # a multiply-by-zero, so I added a BF_MULTIPLY function to do just that.
    # A more sophisticated multiplication system or a human could make this 
    # short circut without the if-else block in BF_MULTIPLY
    #return ( MEM * RNRM(CMP8(MEM_READ(MEM,  STACK_P), 0)).real ) + ( SCAN_RIGHT(MEM) * RNRM(NOT(CMP8(MEM_READ(MEM,  STACK_P), 0))).real )  
    return ( (BF_MULTIPLY(partial(        ID,MEM), int(RNRM(    CMP8(MEM_READ(MEM,  STACK_P), 0) ).real)) )  
           + (BF_MULTIPLY(partial(SCAN_RIGHT,MEM), int(RNRM(NOT(CMP8(MEM_READ(MEM,  STACK_P), 0))).real)) )
           )
           
           
def LEFT_BRACKET(MEM):
    # If the value at *BF_P != 0 then leave everything alone in anticipation of PC++
    ### TODO: CHANGE CMP16 to a bigger CMP if needed ###
    return ( (BF_MULTIPLY(partial(        ID,MEM), int(RNRM(NOT(CMP16(MEM_READ(MEM,  MEM_READ(MEM,  BF_P)), 0))).real)) )
           + (BF_MULTIPLY(partial(SCAN_RIGHT,MEM), int(RNRM(    CMP16(MEM_READ(MEM,  MEM_READ(MEM,  BF_P)), 0) ).real)) )
           )

def OUTPUT(MEM):
    print(f"{chr(int(MEM_READ(MEM,  MEM_READ(MEM,  BF_P))))}", end='')
    return MEM

def INPUT(MEM):
    MEM = MEM_WRITE(MEM, MEM_READ(MEM,  BF_P), ord(getch()) )
    return MEM

def STEP(MEM):
    MEM =  ( BF_MULTIPLY(  partial(    PTR_RIGHT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PR) ).real)  )
           + BF_MULTIPLY(  partial(     PTR_LEFT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PL) ).real)  )
           + BF_MULTIPLY(  partial(      PTR_INC,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PI) ).real)  )
           + BF_MULTIPLY(  partial(      PTR_DEC,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PD) ).real)  )
           + BF_MULTIPLY(  partial( LEFT_BRACKET,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), LB) ).real)  )
           + BF_MULTIPLY(  partial(RIGHT_BRACKET,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), RB) ).real)  )
           + BF_MULTIPLY(  partial(       OUTPUT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), OP) ).real)  )
           + BF_MULTIPLY(  partial(        INPUT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), IP) ).real)  )
           )
    MEM = MEM_WRITE(MEM, PROG_P, MEM_READ(MEM, PROG_P) + 1)
    return RUN(MEM)
            

def RUN(MEM):
    return ( BF_MULTIPLY(  partial(  ID,MEM), int(RNRM(    CMP16(MEM_READ(MEM, PROG_P), MEM_READ(MEM, BF_HALT)) ).real)  )
           + BF_MULTIPLY(  partial(STEP,MEM), int(RNRM(NOT(CMP16(MEM_READ(MEM, PROG_P), MEM_READ(MEM, BF_HALT)))).real)  )
           )

   
#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# These are **ALMOST** legit.  I'm using a control block 
# to deal with python's recursion limit; If python had 
# Tail-call optimization, then this would be unneeded
# Again, if done by hand, these optimizations would be implicit

def STEP_TCO(MEM):
    # from ipdb import set_trace; set_trace()
    MEM =  ( BF_MULTIPLY(  partial(    PTR_RIGHT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PR) ).real)  )
           + BF_MULTIPLY(  partial(     PTR_LEFT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PL) ).real)  )
           + BF_MULTIPLY(  partial(      PTR_INC,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PI) ).real)  )
           + BF_MULTIPLY(  partial(      PTR_DEC,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), PD) ).real)  )
           + BF_MULTIPLY(  partial( LEFT_BRACKET,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), LB) ).real)  )
           + BF_MULTIPLY(  partial(RIGHT_BRACKET,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), RB) ).real)  )
           + BF_MULTIPLY(  partial(       OUTPUT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), OP) ).real)  )
           + BF_MULTIPLY(  partial(        INPUT,MEM), int(RNRM( CMP8(MEM_READ(MEM, MEM_READ(MEM, PROG_P)), IP) ).real)  )
           )
    MEM = MEM_WRITE(MEM, PROG_P, MEM_READ(MEM, PROG_P) + 1)
    # bf_print(MEM)
    return MEM

def RUN_TCO(MEM): 
    """
    While I could increase the python recursion depth, this while-block 
    works around it.  see RUN function above for pure execution
    """
    # bf_print(MEM)
    while int(RNRM(NOT(CMP16(MEM_READ(MEM, PROG_P), MEM_READ(MEM, BF_HALT)))).real):
        MEM = ( BF_MULTIPLY(  partial(      ID,MEM), int(RNRM(    CMP16(MEM_READ(MEM, PROG_P), MEM_READ(MEM, BF_HALT)) ).real)  )
              + BF_MULTIPLY(  partial(STEP_TCO,MEM), int(RNRM(NOT(CMP16(MEM_READ(MEM, PROG_P), MEM_READ(MEM, BF_HALT)))).real)  )
              )
    # bf_print(MEM)
    return MEM
     
#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Machine state displays

from pprint import pprint
def bf_print(MEM):
    print("===")
    print(f"  PC: {MEM[0][1]}")
    print(f"  SC: {MEM[1][1]}")
    print(f"HALT: {MEM[2][1]}")
    print(f" BFP: {MEM[3][1]}")
    print("---")
    print(f"Program : {''.join( chr(int(x[1])) for x in MEM[4:int(MEM[2][1])])} ")
    print(f"address : {''.join(f'{x[0]:2}'[-2] for x in MEM[4:int(MEM[2][1])])} ")
    print(f"          {''.join(f'{x[0]:2}'[-1] for x in MEM[4:int(MEM[2][1])])} ")
    print("---")
    pprint(MEM[int(MEM[2][1]):], compact=True)
    print("===")

#-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

    # bf_print(MEM_REFINE(BF_MEM)) # Display starting memory
    # BF_MEM = RUN(BF_MEM)
    # bf_print(MEM_REFINE(BF_MEM)) # Display the ending memory
    
    print("The Moment of truth: Hello program.")
    LOAD_PROGRAM("++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.")
    RUN_TCO(BF_MEM)

"""    
    MEM_MAT = [[ 0, 0]
              ,[ 1, 1]
              ,[ 2, 2]
              ,[ 3, 3]
              ]
    print(MEM_MAT)
    MEM_MAT = MEM_WRITE(MEM_MAT,2,99,bitwidth=4)
    print(MEM_MAT)

    MEM_MAT = MEM_WRITE(MEM_MAT,1,99,bitwidth=4)
    print(MEM_READ(MEM_MAT, 0))
    print(MEM_READ(MEM_MAT, 1))
    print(MEM_READ(MEM_MAT, 2))
    print(MEM_READ(MEM_MAT, 3))

    
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

    for a in [f,t]: 
     for b in [f,t]: 
      for c in [f,t]:
        print
        x = IMGL(a*b*t)**2
        y = IMGL(a*c*t)**2
        z = IMGL(a * IMGL(b*c) * t)**2
        print "T ?= IMGL(xyU)/z = {}\twhere a={}, b={}, c={}, x={}, y={}, z={}".format(
                IMGL(x*y*u)/z, dispU(a), dispU(b), dispU(c), x, y, z)
        #print "OR(AND(a,b),AND(a,c)) = {}".format(dispU(OR(AND(a,b),AND(a,c))))
        #print "AND(a, OR(b,c)) = {}".format(dispU(AND(a, OR(b,c))))
        #print
        #print "OR(AND(a,b),AND(a,c)) = {}".format(dispU(OR(AND(a,b),AND(a,c))))
        #print "IMGL(AND(a,b) * AND(a,c)) = {}".format(dispU(IMGL(AND(a,b) * AND(a,c))))
        #print "IMGL(NOT(IMGL(a*b*t)) * AND(a,c)) = {}".format(dispU( IMGL(NOT(IMGL(a*b*t)) * AND(a,c)) ))
        #print "IMGL(NOT(IMGL(a*b*t)) * NOT(IMGL(a*c*t))) = {}".format(dispU( IMGL(NOT(IMGL(a*b*t)) * NOT(IMGL(a*c*t))) ))
        #print "IMGL((IMGL(a*b*t)**2 * t) * (IMGL(a*c*t)**2 * t)) = {}".format(dispU( IMGL((IMGL(a*b*t)**2 * t) * (IMGL(a*c*t)**2 * t)) ))
        #print "IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) = {}".format(dispU( IMGL((IMGL(a*b*t)**2 * t) * (IMGL(a*c*t)**2 * t)) ))
        #print
        #print "AND(a, OR(b,c)) = {}".format(dispU(AND(a, OR(b,c))))
        #print "AND(a, IMGL(b*c)) = {}".format(dispU( AND(a, IMGL(b*c)) ))
        #print "NOT(IMGL(a * IMGL(b*c) * t)) = {}".format(dispU( NOT(IMGL(a * IMGL(b*c) * t)) ))
        #print "(IMGL(a * IMGL(b*c) * t))**2 * t = {}".format(dispU( (IMGL(a * IMGL(b*c) * t))**2 * t ))
        #print
        print "IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 ?= t {}".format(dispU( IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 ))
        ##print "IMGL(IMGL(a*t*IMGL(b*c))**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 ?= t {}".format(dispU( IMGL(IMGL(a*t*IMGL(b*c))**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 ))
        #print
        #print "IMGL(a*t) == IMGL(IMGL(a)*t) ? {}".format( IMGL(a*t) == IMGL(IMGL(a)*t) )
        #print "IMGL(a*u) == IMGL(IMGL(a)*u) ? {}".format( IMGL(a*u) == IMGL(IMGL(a)*u) )
        #print "IMGL(a*f) == IMGL(IMGL(a)*f) ? {}".format( IMGL(a*f) == IMGL(IMGL(a)*f) )
        print
        ##print "IMGL(a*b*t) == IMGL(IMGL(a*b)*t) ? {}".format( RIML(a*b*t) == RIML( RIML(a*b)*t) )
        ##print "IMGL(a*b*u) == IMGL(IMGL(a*b)*u) ? {}".format( RIML(a*b*u) == RIML( RIML(a*b)*u) )
        ##print "IMGL(a*b*f) == IMGL(IMGL(a*b)*f) ? {}".format( RIML(a*b*f) == RIML( RIML(a*b)*f) )
        # print "IMGL(a*b*t) == IMGL(IMGL(a*b)*t) ? {}".format( IMGL(a*b*t)**2 == IMGL(IMGL(a*b)**2 *t) )
        print "IMGL(IMGL(a*b*t)**2 * IMGL(a*b*t)**2 * u) / (IMGL(a * IMGL(a*b) * t))**2 == t {}".format(REFINE( IMGL(IMGL(a*b*t)**2 * IMGL(a*b*t)**2 * u) / (IMGL(a * IMGL(a*b) * t))**2 ) == t )
        # print "IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 == t {}".format(dispU( IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / (IMGL(a * IMGL(b*c) * t))**2 ))
        # print "IMGL(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / (IMGL(a * IMGL(b*c) * t)**2 * t) == f {}".format( RIML(IMGL(a*b*t)**2 * IMGL(a*c*t)**2 * u) / REFINE(IMGL(a * IMGL(b*c) * t)**2 * t) == f )
"""
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
