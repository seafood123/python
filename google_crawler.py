from icrawler.builtin import GoogleImageCrawler
import csv


def google_crawler(CSV,MAX_NUM=1000):
    f = open(CSV)
    rdr = csv.reader(f)
    query_list = []
    for list in rdr:
        query_list.extend(list)
    for i in range(len(query_list)):
        query = query_list[i]
        query = str(query)

        google_crawler = GoogleImageCrawler(parser_threads=2,downloader_threads=4,
                                            storage = {'root_dir':query})
        google_crawler.crawl(keyword = query,max_num=MAX_NUM,date_min=None,date_max=None,
                             min_size=(200,200),max_size=None)
    f.close()

if __name__ == "__main__":
    google_crawler("C://Users/abc/Desktop/data.csv",10)

