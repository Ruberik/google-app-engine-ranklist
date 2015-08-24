#!/usr/bin/python
#
# Copyright 2008 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from google.appengine.api import datastore
from google.appengine.api import datastore_errors
from google.appengine.api import datastore_types
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
from ranker import ranker

def GetRanker():
  key = datastore_types.Key.from_path("app", "default")
  try:
    return ranker.Ranker(datastore.Get(key)["ranker"])
  except datastore_errors.EntityNotFoundError:
    r = ranker.Ranker.Create([0, 10000], 100)
    app = datastore.Entity("app", name="default")
    app["ranker"] = r.rootkey
    datastore.Put(app)
    return r

def ReturnError(self, error):
  template_values = {"error": error}
  path = os.path.join(os.path.dirname(__file__), 'error.html')
  self.response.out.write(template.render(path, template_values))

class MainPage(webapp.RequestHandler):
  def get(self):
    path = os.path.join(os.path.dirname(__file__), 'index.html')
    self.response.out.write(template.render(path, {}))


class SetScoreHandler(webapp.RequestHandler):
  def get(self):
    score = self.request.get("score")
    name = self.request.get("name")
    try:
      assert len(name) > 0
      assert not (name[0] in "0123456789")
      score = int(score)
      assert 0 <= score <= 9999
    except:
      ReturnError(self, "Your name must not be empty, and must not start with "
                  "a digit.  In addition, your score must be an integer "
                  "between 0 and 9999, inclusive.")
      return
    r = GetRanker()
    r.SetScore(name, [score])
    self.redirect("/")


class QueryRankPage(webapp.RequestHandler):
  def get(self):
    r = GetRanker()
    rank = int(self.request.get("rank"))
    if rank >= r.TotalRankedScores():
      ReturnError(self, "There aren't %d ranked people!" % (rank + 1))
    else:
      (score, rank_at_tie) = r.FindScore(rank)
      template_values = {"score": score[0], "rank": rank}
      if rank_at_tie < rank:
        template_values["tied"] = True
        template_values["rank_at_tie"] = rank_at_tie

      path = os.path.join(os.path.dirname(__file__), 'rank.html')
      self.response.out.write(template.render(path, template_values))


class QueryScorePage(webapp.RequestHandler):
  def get(self):
    r = GetRanker()
    try:
      score = int(self.request.get("score"))
      assert 0 <= score <= 9999
    except:
      ReturnError(self, "Scores must be integers between 0 and 9999 "
                  "inclusive.")
      return
    rank = r.FindRank([score])
    template_values = {"score": score, "rank": rank}

    path = os.path.join(os.path.dirname(__file__), 'score.html')
    self.response.out.write(template.render(path, template_values))


application = webapp.WSGIApplication(
                                     [('/', MainPage),
                                      ('/setscore', SetScoreHandler),
                                      ('/getrank', QueryRankPage),
                                      ('/getscore', QueryScorePage)
                                      ],
                                     debug=True)

def main():
  run_wsgi_app(application)

if __name__ == "__main__":
  main()
