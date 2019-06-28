# CS573 HW5 - K-Means clustering and Hierarchical Clustering
### Author: Andres Alejos

---

## Python Files and How to Call Them:
    - exploration.py : python exploration.py
    - kmeans.py : python kmeans.py [dataFilename] [# of clusters]
    - kmeans_analysis.py : python kmeans_analysis.py [dataFilename]
      - Performs all of the functions asked for in section 2 of the handout
      - I did not create a separate file for each subsection
    - hierarchical.py : python hierarchical.py [dataFilename]
      - Performs all of the functions asked for in section 3 of the handout
      - I did not create a separate file for each subsection
    - progressBar.py: Not called from the command line.  Used across entire homework to show training progress

## Notes:
    - I have included all of the figures, including the raw_digits images
    - I used the definitions for all of the metrics (WC-SSD, SC, NMI) from the slides
    - In my write up, I used a whole page for each figure so that each would be legible
    - For Part 3.3, where we were instructed to vary the choice of k, I used the same k-values as in Part 2 ([2,4,8,16,32]) so that I could compare consistently to my k-means results
