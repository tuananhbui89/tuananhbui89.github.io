# bundle exec jekyll serve --draft
bundle exec jekyll build --lsi
rsync -av --progress _site/ ../tuananhbui89.github.io
cd ../tuananhbui89.github.io
git add .
git commit -m "update"
git push