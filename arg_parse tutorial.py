# computing volume of cylinder
import math
import argparse

# container to hold our arguments
# metavar - cleans the output obtained after -h (help)
parser = argparse.ArgumentParser(description='Calculate volume of cylinder')
parser.add_argument('-r','--radius',type = int,metavar = '',required= True,help='Radius of Cylinder')
parser.add_argument('-H','--height',type = int,metavar = '',required= True,help='Height of Cylinder')

# mutually exclusive arguments
# we cannot use more than 1 argument at once
group = parser.add_mutually_exclusive_group()
# action = store_true -> default value becomes false and when we call this flag it will be given value true
group.add_argument('-q','--quiet',action='store_true',help = 'print quiet')
group.add_argument('-v','--verbose',action='store_true',help = 'print verbose')

# parse the arguments added
args = parser.parse_args()

def cylinder_vol(radius,height):
    return (math.pi)*(radius ** 2)*(height)

if __name__ == '__main__':
    volume = cylinder_vol(args.radius,args.height)
    if args.quiet:
        print(volume)
    elif args.verbose:
        print("volume of cylinder:" , volume)
    else:
        print("none volume:" , volume)


