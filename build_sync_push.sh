# bundle exec jekyll serve --draft
bundle exec jekyll build --lsi
rsync -av --progress --delete _site/blog/ ../tuananhbui89.github.io/blog
rsync -av --progress --delete _site/projects/ ../tuananhbui89.github.io/projects
rsync -av --progress _site/ ../tuananhbui89.github.io
cd ../tuananhbui89.github.io
git add .
git commit -m "update"
git push