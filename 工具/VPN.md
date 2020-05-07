# VPN
##### 使用秋水逸冰的脚本搭建VPN
1. root用户登录
2. wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocksR.sh
3. chmod +x shadowsocksR.sh
4. ./shadowsocksR.sh 2>&1 | tee shadowsocksR.log
##### 常用命令
- 启动：/etc/init.d/shadowsocks start
- 停止：/etc/init.d/shadowsocks stop
- 重启：/etc/init.d/shadowsocks restart
- 状态：/etc/init.d/shadowsocks status
- 配置文件路径：/etc/shadowsocks.json
- 日志文件路径：/var/log/shadowsocks.log
- 代码安装目录：/usr/local/shadowsocks
