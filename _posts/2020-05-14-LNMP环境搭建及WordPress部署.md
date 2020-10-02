---
layout: post
#title:  "SNMP"
date: 2020-5-14 18:43:22
tags: Linux
color: rgb(3,101,100)
#subtitle: 'Welcome to Jekyll!'
---
1、添加用户组和用户：

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

由于之前已经安装过mysql和www-data相关组，此处提示已存在。

2、安装nginx。

按照网上教程安装时由于deepin没有yum软件包管理器，故首先安装yum：

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

 

此时发现yum是cent os等Linux发行版的软件包管理，像Ubuntu、deepin这种用的是apt-get，故换另一种方法安装。

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

然后选择安装目录，我选择/usr/local/src：

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image009.png)

首先安装PCRE源码包：

Cd pcre-8.39

./configure

Make

Make install

然后安装zlib源码包：

Cd zlib-1.2.8

./configure

Make

Make install

安装ssl：

Wget `https://www.openssl.org/source/openssl-1.0.1t.tar.gz`

tar -zxvf openssl-1.0.1t.tar.gz

然后安装nginx：

Cd nginx-1.10.1

./configure 

--sbin-path=/usr/local/nginx/nginx 

--conf-path=/usr/local/nginx/nginx.conf 

--pid-path=/usr/local/nginx/nginx.pid 

--with-http_ssl_module 

--with-pcre=/usr/local/src/pcre-8.39 

--with-zlib=/usr/local/src/zlib-1.2.8 

--with-openssl=/usr/local/src/openssl-1.0.1e

Make

到这一步报错了！！！

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image011.jpg)

原因是把警告当成了错误来处理，解决方法如下：

打开nginx/objs/Makefile，去掉-Werror:

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg)

然后重新执行make && make install，完成后运行nginx：

sudo /usr/local/nginx/nginx

打开浏览器，输入localhost：80，出现nginx的欢迎界面，则安装成功。

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image015.jpg)

然后设置使其开机自动启动：

在/etc/init.d中创建nginx文件，输入nginx的初始化命令，保存后给文件执行权限：

![img](file:///C:/Users/WHU-GE~1/AppData/Local/Temp/msohtmlclip1/01/clip_image016.png)

安装sysv-rc-conf：

Sudo apt-get install sysv-rc-conf。

给nginx赋予开机启动的选项。

\3.   安装mysql。

将安装包进行解压：tar -xzvf mysql-5.6.31.tar.gz 

安装依赖库：apt-get -y install make cmake gcc g++ bison openssl libssl-dev libncurses5-dev

复制到/usr/local/mysql目录下：sudo cp -r mysql-5.6.31 /usr/local/mysql，进入此目录进行安装。

Sudo cmake . -DCMAKE_INSTALL_PREFIX=/usr/local/mysql -DMYSQL_DATADIR=/usr/local/mysql/data -DSYSCONFDIR=/etc -DWITH_INNOBASE_STORAGE_ENGINE=1 -DWITH_ARCHIVE_STORAGE_ENGINE=1 -DWITH_BLACKHOLE_STORAGE_ENGINE=1 -DWITH_PARTITION_STORAGE_ENGINE=1 -DWITH_PERFSCHEMA_STORAGE_ENGINE=1 -DWITHOUT_EXAMPLE_STORAGE_ENGINE=1 -DWITHOUT_FEDERATED_STORAGE_ENGINE=1 -DDEFAULT_CHARSET=utf8 -DDEFAULT_COLLATION=utf8_general_ci -DWITH_EXTRA_CHARSETS=all -DENABLED_LOCAL_INFILE=1 -DWITH_READLINE=1 -DMYSQL_UNIX_ADDR=/usr/local/mysql/mysql.sock -DMYSQL_TCP_PORT=3306 -DMYSQL_USER=mysql -DCOMPILATION_COMMENT="lq-edition" -DENABLE_DTRACE=0 -DOPTIMIZER_TRACE=1 -DWITH_DEBUG=1

编译：sudo make

又出现了把警告当错误的情况，老办法解决，想要寻找makefile文件中的cflag，然而多次查询也没找到，故尝试更换教程提供的安装包重新编译，使用mysql-5.7.21安装包，重新操作，顺利编译。

接着make install，添加用户和用户组。

设置安装目录的权限：sudo chown -R mysql:mysql ./

开启ssl功能：bin/mysql_ssl_rsa_setup

```
测试启动：bin/mysqld_safe --user=mysql，然而在这一步出现了如下报错：

查找mysqld进程通过kill进行关闭，然后重试，成功。
启动mysql服务：support-files/mysql.server start。
将mysql服务放在/etc/init.d目录下：cp support-files/mysql.server /etc/init.d/mysql.server。
将mysql添加到环境变量：vim ~/.bashrc，在开头添加：
export MYSQL_HOME=/usr/local/mysql
export PATH=$PATH:$MYSQL_HOME/bin
然后使环境变量生效：source ~/.bashrc。
4.安装php。
首先安装依赖库mcrypt、libxml2、curl：
apt-get install libmcrypt-dev
apt-get install libxml2-dev
apt-get install curl
下载php安装包解压，进入php解压目录下：

进行配置参数的设置:
./configure --prefix=/usr/local \
--with-mcrypt=/usr/include \
--with-mysqli=mysqlnd \
--with-pdo-mysql=mysqlnd \
--with-iconv \
--with-zlib \
--with-openssl \
--with-curl \
--enable-fpm \
--enable-mbstring \
--with-mysql=/usr/share/mysql/
编译安装，注意到最后提示WARNING: unrecognized options: --with-mysql。
进行相关查询后改用 --with-pdo-mysql，重新配置,又出现另一个问题：
error: Unable to find your mysql installation
于是验证路径的正确性：which mysql
得到路径为 /usr/bin/mysql,更改后重新尝试，又出现错误：

暂且跳过这个错误尝试make&&make install，一切正常，测试版本成功显示：

4.至此，nginx、mysql、php安装完成，下面进行相关的配置操作。
5.打开nginx的配置文件nginx.conf，添加对php的支持：


配置php：
将php-fpm添加到服务：cp init.d.php-fpm /etc/init.d/php-fpm
赋予执行权限，通过sysv-rc-conf使其可以开机自动启动。
然后测试是否正确支持php，访问localhost/info.php提示file not found，经过排错发现nginx.conf文件中php部分修改有误，修改为如下：

再次访问，正常运行。
5.安装wordpress。
安装之前要先给wordpress创建数据库：

创建用户刷新权限：
grant all on wordpress.* to ‘whublog’@’localhost’ identified by ‘123456’;
flush privileges;
准备工作到这里全部完成，下面解压wordpress安装包，并将其拷贝到nginx/html目录下：

重命名配置文件：

对文件进行数据库的配置：
将文件wp-config-sample.php里面的数据库相关内容修改为如下所示，并将文件名修改为wp-config.php。

然后配置nginx.conf文件，将root路径改为wordpress文件目录：

然后在浏览器中输入localhost/wp-admin进入后台管理界面：

然后就可以在此界面对博客的主题等组件进行自定义操作并进行发布，在浏览器中访问localhost进入站点：

 
三．安装基于nginx的网络媒体服务+浏览器媒体播放+嵌入至wordpress页面中。
1.查看原有nginx的配置参数：

查看文件发现是没有nginx-rtmp-module模块的，尝试单独安装。
首先从github上获取源码文件放在/usr/local/src目录下。
重新编译nginx，在生成makefile配置文件的参数中加上对rtmp模块的支持：

同样在make之前要去掉-Werror，然而出现了新的问题：
发现是openssl包的版本问题，替换为1.0.1版本，顺利make。
然后将objs目录下的nginx可执行程序替换/usr/local/nginx下的nginx程序，重启nginx服务：service nginx restart。
接着配置nginx.conf，加入如下支持rtmp的代码：
rtmp_auto_push on;
rtmp {       
    server {    
        listen 1935;  #监听的端口      
        chunk_size 4096;
 
        application vod {
            play /home/wwwroot/video; #video store-location
        }  
             
        application hls {  #rtmp推流请求路径  
            live on;    
            hls on;    
            hls_path /home/wwwroot/hls;    
            hls_fragment 5s;    
        }    
        application live{
            live on;        
        }
}   
 

验证推流是否成功，采用VLC media player播放器来进行验证，过程如下：


```

 

 

 

 

 

 

 