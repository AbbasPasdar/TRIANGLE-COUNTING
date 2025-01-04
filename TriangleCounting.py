from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import statistics
import time


def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def HashFunction(u,a,b,c):  # Hash Function with u as edge c as color
    p = 8191
    h = ((a*u+b) % p) % c
    return h


def subset_of_edges(pairs,a,b,C): # Subset extraction of edges using key-value pairs
    E = {}
    for u,v in pairs:
        i=HashFunction(u,a,b,C) # Assigning keys based on hashing
        j=HashFunction(v,a,b,C)
        if i==j:
            if i not in E.keys():
                E[i] = (u,v)
            else:
                E[i].add((u,v))
    return [(key, E[key]) for key in E.keys()]


# Algorithm1
def MR_ApproxTCwithNodeColors(edges,a,b,C):
    t = (edges.flatMap(lambda x:subset_of_edges([x],a,b,C)) # <-- MAP PHASE (R1)
		.groupByKey()                                   # <-- SHUFFLE+GROUPING
        .mapValues(lambda x: CountTriangles(x))         # <-- REDUCE PHASE (R1)
        .map(lambda x: x[1])                            # <-- MAP PHASE (R2)
        .reduce(lambda x, y: x + y))                    # <-- REDUCE PHASE (R2)
    tFinal = t * (C ** 2)
    return tFinal
    
# Algorithm2        
def MR_ApproxTCwithSparkPartitions(edges,C):
    t = (edges.mapPartitions(lambda x: [CountTriangles(x)]) # <-- REDUCE PHASE (R1)
        .reduce(lambda x, y: x + y))                        # <-- REDUCE PHASE (R2)
    tFinal = t * (C ** 2)
    return tFinal

      
    
def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python3 G004HW1.py 1 1 facebook_large.txt"

	# SPARK SETUP
    conf = SparkConf().setAppName('test')
    sc = SparkContext(conf=conf)

	# INPUT READING
	# 1. Reads parameters C and R
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)
    
    R = sys.argv[2]
    assert R.isdigit(), "C must be an integer"
    R = int(R)
    
    # 2. Read input file 
    
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    
    # Reads the input graph into an RDD of strings 
    rawData = sc.textFile(data_path)
    
    # transform the RDD of strings into an RDD of edges
    edges = rawData.map(lambda x: tuple(map(int, x.split(',')))).repartition(numPartitions=C).cache()
    
    # Print the name of the file, R,C, and the number of edges
    numedges = edges.count()
    print("Dataset = ", os.path.basename(rawData.name()), "\nNumber of Edges = ", numedges,"\nNumber of Colors = ", C,"\nNumber of Repetitions = ",R)
    
    #Run Algorithm1
    results = []
    runtimes = []
    p = 8191
    for i in range(R):
        a = rand.randint(0, p-1)  # Changing a,b before calling function
        b = rand.randint(0, p-1)
        start_time = time.time()
        results.append(MR_ApproxTCwithNodeColors(edges,a,b,C))
        end_time = time.time()
        runtime1 = end_time - start_time
        runtimes.append(runtime1)
        
    result1 = statistics.median(results)
    # compute the average execution time
    avg_runtime = round(sum(runtimes)*1000 / R)
    print("Approximation through node coloring")
    print("- Number of triangles (median over", R, "runs) = ", result1)
    print("- Running time (average over", R, "runs) = ", avg_runtime, " ms")
 
    
    #Run Algorithm2
    start_time = time.time()
    result2=MR_ApproxTCwithSparkPartitions(edges,C)
    end_time = time.time()
    runtime2 = end_time - start_time
    runtime2=round(runtime2*1000)
    print("Approximation through Spark partitions")
    print("- Number of triangles = ", result2)
    print("- Running time = ", runtime2, " ms")

if __name__ == "__main__":
    main()
