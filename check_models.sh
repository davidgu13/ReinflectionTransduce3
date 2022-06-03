langs=(
V
A
)

pos=(
  Asia
  Europe
  America
)

for index in ${!langs[*]}; do
  echo "${langs[$index]} - ${pos[$index]}"
done


bash dummy.sh kat V form 100

bash dummy.sh kat N form 100

bash dummy.sh swc V form 100

bash dummy.sh swc ADJ form 100

bash dummy.sh sqi V form 100

bash dummy.sh hun V form 100

bash dummy.sh bul V form 100

bash dummy.sh bul ADJ form 100

bash dummy.sh lav V form 100

bash dummy.sh lav N form 100

bash dummy.sh tur V form 100

bash dummy.sh tur ADJ form 100

bash dummy.sh fin N form 100

bash dummy.sh fin ADJ form 100

bash dummy.sh fin V form 100

