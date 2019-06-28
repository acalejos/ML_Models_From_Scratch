#!/usr/bin/env python
import csv
import sys
#Task 1: Remove unnecessary quotes in columns 'race', 'race_o', and 'field' and report number of cells changed
def strip_quotes(input_file,output_file):
    count = 0
    count2 = 0
    headers1 = ['race','race_o','field']
    headers2 = ['field']
    headers3 = ['gender','race','race_o','field']
    headers4 = ['gaming','reading']
    preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
    preference_scores = preference_scores_of_participant + preference_scores_of_partner
    preference_totals = {}
    preference_counts = {}
    for score in preference_scores:
        preference_totals[score] = 0
        preference_counts[score] = 0
    encoding_values = {}
    for header in headers3:
        encoding_values[header] = []
    for row in input_file:
        out = row
        for header in headers4:
            current = row[header]
            if int(current) > 10:
                out[header] = 10
        for header in headers1:
            current = row[header]
            if current.startswith("'") and current.endswith("'") and len(current) > 1:
                current = current.strip("\'")
                out[header] = current
                count+=1
        for header in headers2:
            current = row[header]
            if current[0].isupper():
                count2 += 1
                current = current.lower()
                out[header] = current
        for header in headers3:
            current = row[header]
            values = encoding_values.get(header)
            if current not in values:
                encoding_values[header] = values + [current]
        for header in headers3:
            encoding_values.get(header).sort()
            current = row[header]
            out[header] = encoding_values.get(header).index(current) 
        for header in preference_scores:
            current = row[header]
            preference_totals[header] += float(current)
            preference_counts[header] += 1
        for header in preference_scores:
            current = row[header]
            if preference_totals[header] != 0:
                out[header] = float(current) / preference_totals[header]
        output_file.writerow(out)
    print("Quotes removed from " + str(count) + " cells.")
    print("Standardized " + str(count2) + " cells to lower case.")
    print("Value assigned for male in column gender: " + str(encoding_values.get('gender').index('male'))+".")
    print("Value assigned for European/Caucasian-American in column race: " + str(encoding_values.get('race').index('European/Caucasian-American'))+".")
    print("Value assigned for Latino/Hispanic American in column race_o: " + str(encoding_values.get('race_o').index('Latino/Hispanic American'))+".")
    print("Value assigned for law in column gender: " + str(encoding_values.get('field').index('law'))+".")
    for header in preference_scores:
        print("Mean of " + header + ": " + str(round(preference_totals[header]/preference_counts[header],2))+".")

#Main function
def main(argv):
    f = open(argv[0])
    o = open(argv[1],"w+")
    input_file = csv.DictReader(f)
    fieldnames = input_file.fieldnames
    output_file = csv.DictWriter(o,fieldnames=fieldnames,lineterminator='\n')
    output_file.writeheader()
    strip_quotes(input_file,output_file)
    f.close()
    o.close()    

if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Incorrect arguments")
        sys.exit(0)
    else:
        main(sys.argv[1:])
