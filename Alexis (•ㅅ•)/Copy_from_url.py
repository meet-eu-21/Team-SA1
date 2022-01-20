import urllib3
http = urllib3.PoolManager()

def copy_from_url(start_url,mid_url,end_url,start_file,end_file,n1,n2):
    for i in range(n1,n2+1):

        r = http.request('GET', start_url+str(i)+mid_url+str(i)+end_url)

        file=open(start_file+str(i)+end_file,"w")
        for j in r.data.decode("utf-8") :
            file.write(str(j))
        print(start_file+str(i)+end_file)
        file.close()


