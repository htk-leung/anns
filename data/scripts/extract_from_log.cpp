#include <fstream>
#include <cctype>

#define INFILEPATH "../logresult/20250924-flatnav-rand.log"
#define OUTFILEPATH "../logresult/EXPORTflatnav.txt"

#define TOTALVAR 9


int main() {
    std::ifstream infile(INFILEPATH, std::ios::in);
    std::ofstream outfile(OUTFILEPATH, std::ios::out);
    
    char c;
    int count = TOTALVAR;

    while(infile.get(c)) {
        if(c == '=' || c == ':') {
            infile.get(c); // space
            if(count != TOTALVAR)
                outfile << ',';
            while(infile.get(c) && !isalpha(c) && !isblank(c) && c!='\n') 
                outfile << c;
            count--;
        }
        if(count == 0) {
            outfile << std::endl;
            count = TOTALVAR;
        }
    }

    outfile.close();
    infile.close();

    return 0;
}