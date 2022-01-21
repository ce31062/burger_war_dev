#!/bin/bash
###############################################################################
#-dockerとdocker-composeをインストールする
#-
#+[USAGE]
#+  $0 [-h] [amd64|armhf|arm64]
#+
#-[OPTIONS]
#-  -h            このヘルプを表示
#-
#-[ARGUMENTS]
#-  amd64         x86_64/amd64環境用のdockerをインストールする
#-  armhf         ARM32bit環境用のdockerをインストールする
#-  arm64         ARM64bit環境用のdockerをインストールする
#-
#-[REFERENCES]
#-  docker  :     https://docs.docker.com/engine/install/ubuntu/
#-  compose :     https://docs.docker.com/compose/install/
#-
###############################################################################
set -e
set -u
CMD_NAME=$(basename $0)

# 設定値
#------------------------------------------------
# docker-compose version
DC_VERSION=$(curl -s https://github.com/docker/compose/releases/latest|sed -e 's@^.*/releases/tag/\(.*\)">.*$@\1@')


# 関数定義
#------------------------------------------------
usage_exit() {
  # ファイル冒頭のコメントからUSAGEを出力
  sed '/^[^#]/q' "$0"             \
  | sed -n '/^#+/s/^#+//p'        \
  | sed "s/\$0/${CMD_NAME}/g"     1>&2
  exit 1
}
help_exit() {
  # ファイル冒頭のコメントからヘルプを出力
  sed '/^[^#]/q' "$0"             \
  | sed -n '/^#[-+]/s/^#[-+]//p'  \
  | sed "s/\$0/${CMD_NAME}/g"     1>&2
  exit 0
}
err_exit(){
  echo $1
  exit 1
}

# オプション・引数解析
#------------------------------------------------
while getopts h OPT
do
  case $OPT in
    h  ) # ヘルプの表示
      help_exit
      ;;
    \? ) # 不正オプション時のUSAGE表示
      usage_exit
      ;;
  esac
done
shift $((OPTIND - 1))

ARCH=${1:-amd64}
case ${ARCH} in
  amd64 | armhf | arm64 ) ;;
  * ) echo "${ARCH} is invalid architecture!"
      echo "You must specified amd64 or armhf or arm64"
      exit 1
esac


# dockerのインストール
#------------------------------------------------
if !(type "docker" > /dev/null 2>&1); then
  # when not installed docker
  echo "#------------------------------------------------------------------"
  echo "# ${ARCH}用のdockerをインストールします"
  echo "#------------------------------------------------------------------"
  # Update the apt package index and install packages
  sudo apt-get update
  sudo apt-get install -y \
      apt-transport-https \
      ca-certificates \
      curl \
      gnupg-agent \
      software-properties-common


  # Add Docker’s official GPG key
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo apt-key fingerprint 0EBFCD88 \
      | grep "9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88" \
      || err_exit "apt-key verify failed"

  # Use the following command to set up the stable repository
  sudo add-apt-repository \
      "deb [arch=${ARCH}] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"

  #--------------------------------------
  # Install Docker Engine
  #--------------------------------------
  # Install the latest version of Docker Engine and containerd
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io

  # Add user group docker
  sudo usermod -aG docker ${USER}
else
  echo "#------------------------------------------------------------------"
  echo "# dockerがインストール済みのため、処理をスキップしました"
  echo "#------------------------------------------------------------------"
fi

# docker-composeのインストール
#------------------------------------------------
if !(type "docker-compose" > /dev/null 2>&1); then
  echo "#------------------------------------------------------------------"
  echo "# docker-composeをインストールします"
  echo "#------------------------------------------------------------------"
  sudo curl -L "https://github.com/docker/compose/releases/download/${DC_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
else
  echo "#------------------------------------------------------------------"
  echo "# docker-composeがインストール済みのため、処理をスキップしました"
  echo "#------------------------------------------------------------------"
fi

