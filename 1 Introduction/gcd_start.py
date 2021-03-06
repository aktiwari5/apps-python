# Find the greatest common denominator of two numbers
# using Euclid's algorithm


def gcd(a, b):

    if a < a:
        (a, b) = (b, a)
    
    while (a % b) != 0:
        (a, b) = (b, (a%b))

    return(b)

        
        
# try out the function with a few examples
print(gcd(60, 96))  # should be 12
print(gcd(20, 8))   # should be 4
