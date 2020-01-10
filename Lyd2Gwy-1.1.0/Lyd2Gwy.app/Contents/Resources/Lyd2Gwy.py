import parseSTM as ps
import sys, os

if __name__ == "__main__":
    fpaths = []
    print("Please drag STM file(s) into Lyd2Gwy window..")
    for ind, arg in enumerate(sys.argv[1:]):
        if '-h' == arg or '--help' == arg:
            sys.exit("Syntax: lyd2gwy [lydSTMfile1] [lydSTMfile2] ...")
        fpaths.append(arg)

    for path in fpaths:
        gwypath = os.path.split(path)[1]+".gwy"
        if os.path.exists(gwypath):
            user_input = input("%s exists. Do you want to overwrite it? (y/n) " % gwypath)
            if user_input == 'n':
                continue
        data = ps.STMfile(path)
        try:
            data.toGwy()
            print("Succesfully converted %s to gwy file" % os.path.split(path)[1])
        except:
            print("Failed to convert %s to gwy file" % os.path.split(path)[1])
