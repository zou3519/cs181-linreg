import csv
import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print "Usage: python csv2t7.py in.csv train_out.t7"
        return

    filein = sys.argv[1]
    train_out = sys.argv[2]
    #target_out = sys.argv[3]

    print "==> Reading tmp csv"
    reader = csv.reader(open(filein, 'rb'))
    writer = csv.writer(open('~tmp.csv', 'wb'), delimiter=',')
    #writer2 = csv.writer(open('~tmp2.csv','wb'),delimiter=',')

    targets = []

    nrows = 0
    for row in reader:
        if nrows != 0:
            writer.writerow(row[1:])
            #writer2.writerow([ float(row[257]) ])
            #targets.append(float(row[257]))
        #if nrows != len(targets):
        #    print "Warning, something was skipped"
        nrows += 1

    # write one row
    #writer2.writerow(targets)

    print "==> Calling csv2t7.sh"
    subprocess.call(['./csv2t7.sh', '~tmp.csv', train_out] )
    #subprocess.call(['./csv2t7.sh', '~tmp2.csv', target_out] )


    print "==> Cleaning up"
    subprocess.call(['rm', '~tmp.csv'])
    #subprocess.call(['rm', '~tmp2.csv'])

if __name__ == '__main__':
    main()
