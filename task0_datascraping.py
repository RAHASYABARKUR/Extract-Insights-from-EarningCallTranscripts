#GEID 1011139629
#Nimit Shah

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

def getdivs(url) :
    page = requests.get(url)
    #page
    soup = bs(page.content, "html.parser")
    divs = soup.find_all('div')
    return divs



def get_transcript_from_div(parent_div):
    div = parent_div[0]
    lines = div.text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    # [print(line) for line in non_empty_lines]
    l = len(non_empty_lines)
    copyflag = False
    adstr = 'googletag.cmd.push'
    text_start = 'Prepared Remarks:'
    adindex = 0
    ind = 0
    adcounter = 0
    res = ''
    # print(adcounter)
    # for idx, line in enumerate(non_empty_lines) :
    while ind < l:
        line = non_empty_lines[ind]
        # print(line)
        # print('-----------')
        if ind != l - 1 and adstr in non_empty_lines[ind + 1]:
            # print('here', adcounter, line)
            if adcounter == 2:
                # print('breaks')
                break
            else:
                adcounter += 1
            ind = ind + 3
        if text_start in line:
            # print(line)
            copyflag = True
        if copyflag:
            # print(line)
            res = res + line

        ind += 1
    return res

    # print('\nx*x*x*x*x*x*x*x\n')
# f.close()

apple = ["https://www.fool.com/earnings/call-transcripts/2020/04/30/apple-inc-aapl-q2-2020-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2020/01/28/apple-inc-aapl-q1-2020-earnings-call-transcript.aspx",
        "https://www.fool.com/earnings/call-transcripts/2019/10/30/apple-inc-aapl-q4-2019-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/30/apple-inc-aapl-q3-2019-earnings-call-transcript.aspx",
         "https://www.fool.com/earnings/call-transcripts/2019/04/30/apple-inc-aapl-q2-2019-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/01/29/apple-inc-aapl-q1-2019-earnings-conference-call-tr.aspx",
         "https://www.fool.com/earnings/call-transcripts/2018/11/01/apple-inc-aapl-q4-2018-earnings-conference-call-tr.aspx", "https://www.fool.com/earnings/call-transcripts/2018/07/31/apple-inc-aapl-q3-2018-earnings-conference-call-tr.aspx",
         "https://www.fool.com/earnings/call-transcripts/2018/05/01/apple-inc-aapl-q2-2018-earnings-conference-call-tr.aspx", "https://www.fool.com/earnings/call-transcripts/2018/02/01/apple-inc-appl-q1-2018-earnings-conference-call-tr.aspx"]
google = ["https://www.fool.com/earnings/call-transcripts/2020/04/29/alphabet-inc-goog-googl-q1-2020-earnings-call-tran.aspx", "https://www.fool.com/earnings/call-transcripts/2020/02/03/alphabet-inc-goog-googl-q4-2019-earnings-call-tran.aspx",
         "https://www.fool.com/earnings/call-transcripts/2019/10/29/google-inc-googl-q3-2019-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/25/alphabet-inc-googl-q2-2019-earnings-call-transcrip.aspx",
         "https://www.fool.com/earnings/call-transcripts/2019/04/29/alphabet-inc-googl-q1-2019-earnings-call-transcrip.aspx", "https://www.fool.com/earnings/call-transcripts/2019/02/04/alphabet-inc-goog-googl-q4-2018-earnings-conferenc.aspx",
          "https://www.fool.com/earnings/call-transcripts/2018/10/26/alphabet-inc-goog-googl-q3-2018-earnings-conferenc.aspx", "https://www.fool.com/earnings/call-transcripts/2018/02/01/alphabet-inc-googl-q4-2017-earnings-conference-cal.aspx",
         "https://www.fool.com/earnings/call-transcripts/2017/10/30/alphabet-inc-goog-googl-q3-2017-earnings-conferenc.aspx"]
goldman = ["https://www.fool.com/earnings/call-transcripts/2019/07/16/goldman-sachs-group-inc-gs-q2-2019-earnings-call-t.aspx", "https://www.fool.com/earnings/call-transcripts/2019/04/15/goldman-sachs-group-inc-gs-q1-2019-earnings-call-t.aspx",
          "https://www.fool.com/earnings/call-transcripts/2019/01/16/goldman-sachs-group-inc-gs-q4-2018-earnings-confer.aspx", "https://www.fool.com/earnings/call-transcripts/2018/10/16/goldman-sachs-group-inc-gs-q3-2018-earnings-confer.aspx",
          "https://www.fool.com/earnings/call-transcripts/2018/07/17/goldman-sachs-group-inc-gs-q2-2018-earnings-confer.aspx"]
jpmc = ["https://www.fool.com/earnings/call-transcripts/2020/04/14/jpmorgan-chase-co-jpm-q1-2020-earnings-call-transc.aspx", "https://www.fool.com/earnings/call-transcripts/2020/01/14/jpmorgan-chase-co-jpm-q4-2019-earnings-call-transc.aspx",
       "https://www.fool.com/earnings/call-transcripts/2019/10/16/jpmorgan-chase-jpm-q3-2019-earnings-call-transcrip.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/16/jpmorgan-chase-jpm-q2-2019-earnings-call-transcrip.aspx",
       "https://www.fool.com/earnings/call-transcripts/2019/04/12/jpmorgan-chase-co-jpm-q1-2019-earnings-call-transc.aspx", "https://www.fool.com/earnings/call-transcripts/2019/01/15/jpmorgan-chase-co-jpm-q4-2018-earnings-conference.aspx",
       "https://www.fool.com/earnings/call-transcripts/2018/10/12/j-p-morgan-chase-co-jpm-q3-2018-earnings-conferenc.aspx", "https://www.fool.com/earnings/call-transcripts/2018/07/13/j-p-morgan-chase-co-jpm-q2-2018-earnings-conferenc.aspx",
       "https://www.fool.com/earnings/call-transcripts/2018/04/13/j-p-morgan-chase-co-jpm-q1-2018-earnings-conferenc.aspx", "https://www.fool.com/earnings/call-transcripts/2018/01/15/jpmorgan-chase-jpm-q4-2017-earnings-conference-cal.aspx"]
ms = ["https://www.fool.com/earnings/call-transcripts/2020/04/16/morgan-stanley-ms-q1-2020-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/04/17/morgan-stanley-ms-q1-2019-earnings-conference-call.aspx",
     "https://www.fool.com/earnings/call-transcripts/2019/01/17/morgan-stanley-ms-q4-2018-earnings-conference-call.aspx", "https://www.fool.com/earnings/call-transcripts/2018/10/16/morgan-stanley-ms-q3-2018-earnings-conference-call.aspx",
     "https://www.fool.com/earnings/call-transcripts/2018/07/20/morgan-stanley-ms-q2-2018-earnings-conference-call.aspx", "https://www.fool.com/earnings/call-transcripts/2018/04/18/morgan-stanley-ms-q1-2018-earnings-conference-call.aspx"]
citi = ["https://www.fool.com/earnings/call-transcripts/2020/04/15/citigroup-inc-c-q1-2020-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2020/01/14/citigroup-inc-c-q4-2019-earnings-call-transcript.aspx",
       "https://www.fool.com/earnings/call-transcripts/2019/10/15/citigroup-inc-c-q3-2019-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/15/citigroup-inc-c-q2-2019-earnings-call-transcript.aspx",
       "https://www.fool.com/earnings/call-transcripts/2019/04/15/citigroup-inc-c-q1-2019-earnings-call-transcript.aspx", "https://www.fool.com/earnings/call-transcripts/2019/01/15/citigroup-c-q4-2018-earnings-conference-call-trans.aspx",
       "https://www.fool.com/earnings/call-transcripts/2018/10/12/citigroup-inc-c-q3-2018-earnings-conference-call-t.aspx", "https://www.fool.com/earnings/call-transcripts/2018/07/16/citigroup-inc-c-q2-2018-earnings-conference-call-t.aspx",
       "https://www.fool.com/earnings/call-transcripts/2018/04/13/citigroup-inc-c-q1-2018-earnings-conference-call-t.aspx", "https://www.fool.com/earnings/call-transcripts/2018/01/17/citigroup-c-q4-2017-earnings-conference-call-trans.aspx"]
creditsuisse = ["https://www.fool.com/earnings/call-transcripts/2020/04/23/credit-suisse-group-ag-cs-q1-2020-earnings-call-tr.aspx", "https://www.fool.com/earnings/call-transcripts/2020/02/13/credit-suisse-group-ag-cs-q4-2019-earnings-call-tr.aspx",
               "https://www.fool.com/earnings/call-transcripts/2019/11/09/credit-suisse-group-ag-cs-q3-2019-earnings-call-tr.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/31/credit-suisse-group-ag-cs-q2-2019-earnings-call-tr.aspx"]
deutschebank = ["https://www.fool.com/earnings/call-transcripts/2020/04/29/deutsche-bank-ag-db-q1-2020-earnings-call-transcri.aspx", "https://www.fool.com/earnings/call-transcripts/2020/01/30/deutsche-bank-ag-db-q4-2019-earnings-call-transcri.aspx",
               "https://www.fool.com/earnings/call-transcripts/2019/11/08/deutsche-bank-ag-db-q3-2019-earnings-call-transcri.aspx", "https://www.fool.com/earnings/call-transcripts/2019/07/24/deutsche-bank-aktiengesellschaft-db-q2-2019-earnin.aspx"]
all_companies = []
all_companies.append(apple)
all_companies.append(google)
all_companies.append(goldman)
all_companies.append(jpmc)
all_companies.append(ms)
all_companies.append(citi)
all_companies.append(creditsuisse)
all_companies.append(deutschebank)
all_companies
print(all_companies[0])
all_companies_names = ['apple', 'google', 'goldmansachs', 'jpmc', 'morganstanley', 'citicorp', 'creditsuisse', 'deutschebank']

import os

# print(all_companies[0])
for i, company in enumerate(all_companies):
    os.mkdir(all_companies_names[i])
    os.chdir(all_companies_names[i])
    for url in company:
        print(url)
        d = getdivs(url)
        f = get_transcript_from_div(d)
        fname = url.split('/')[-1].split('.')[0] + '.txt'
        fh = open(fname, "w+")
        fh.write(f)
        fh.close()
    os.chdir('..')

