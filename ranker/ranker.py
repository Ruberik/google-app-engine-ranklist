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

from google.appengine.api import datastore
from google.appengine.api import datastore_types

from common import transactional


class Ranker(object):
  """A data structure for storing integer scores and quickly retrieving their
  relative ranks.

  Scores are stored as "name: score" mapping. A score is inserted by
  calling SetScore with a new name. The score can later be updated by
  calling SetScore again with the same name.

  The scores are actually lists of integers with the same number of elements,
  and their ordering is lexicographic.  That is to say that score A is higher
  than score B if they are different, and the first element that differs between
  the two is higher in A.  Thus [5, 3] is ranked higher than [4, 99].

  Example Use Case:

  Some number of people are participating in a programming contest.  Solving a
  problem gets points; contestants also get a tie-breaking "penalty time."  The
  higher the time, the worse the score.

  STEP 1:
  # Creates the ranker when the contest is created:
  rank = ranker.Ranker.Create([0, 10000, -999999, 1], 100)
  # In our contest, people won't have more than 9999 points or 999999 penalty
  # seconds.  Since penalty time is bad, we store [score, -penalty].
  contest['ranker'] = rank.rootkey
  datastore.Put(contest)

  STEP 2:
  # Someone registers for the contest.  The default score is [0, 0], and for
  # efficiency we don't put them in the ranker; they won't be ahead of anyone
  # anyway.

  STEP 3:
  # Player "Jon" gets points for the first time!
  rank = ranker.Ranker(contest['ranker'])
  # Loads up the ranker.  This is the first step of all the STEPs below.
  rank.SetScore("Jon", [5, -120])  # 5 points, 2 minutes

  STEP 4:
  # Player "Jon" gets points for the second time!
  rank.SetScore("Jon", [10, -300])  # 10 points, 5 minutes

  STEP 5:
  # What is the rank of the person with score [10, -300]?
  position = rank.FindRank([10, 300])
  # What are the ranks for the people with scores a, b, c?
  positions = rank.FindRanks([a, b, c])

  STEP 6:
  # What is the score of the person ranked 20th?
  score = rank.FindScore(19)
  # This is particularly useful for seeing multiple pages of ranked people.  If
  # the scores are separately tracked in an entity called 'scores', the
  # datastore can efficiently answer:
  # q = datastore.Query('scores')
  # q['score <='] = rank.FindScore(1000)[0]
  # next_twenty_scores = q.Get(20)
  # This is a simplified case, where scores are a single integer.

  STEP 7:
  # How many people have points?
  num_pointy_people = rank.TotalRankedScores()



  Implementation Details:

  The Ranker is a rooted tree of 'ranker_node' entities.  It is an
  N-ary tree, where N = self.branching_factor, which is specified by the
  constructor.  Each node represents some range of scores, and is assigned a
  unique node_id.

  Take for example a 3-ary tree with point range [0, 10000, -36000, 1].  This
  could represent a contest where contestants can have between 0 and 9999
  points, with ties being broken by (negative) penalty seconds that can be
  between 0 and 10 hours.

  The root represents the score range [0, 10000, -36000, 1].  It has node_id 0.
  Its first child has node_id 1 and score range [0, 3333, -36000, 1].
  The root's second child, node_id 2, has score range [3333, 6666, -36000, 1],
  and its third child, node_id 3, has score range [6666, 10000, -36000, 1].
  Node 1's first child, node_id 4, has range [0, 1111, -36000, 1], and so on.
  See __WhichChild for details of how children are assigned score ranges.

  The point of a node is to track how many stored scores are in the score range
  of each of its children.  The root in the above example would start off with
  child_node_counts = [0, 0, 0]; adding score [4000, 0] would change the root's
  child_node_counts to [0, 1, 0] and node 5's child_node_counts to [1, 0, 0],
  and so forth.

  Ranker also has a "ranker_score" entity for every score stored in the ranker.
  These entities are part of the same entity group as the ranker_node
  entities. This allows for atomic, idempotent calls to SetScores.

  Ranker supports the following operations, which can be read about in detail
  in their docstrings:

  SetScores(scores): Set scores for multiple players.
  FindRank(score): Finds the 0-based rank of the provided score.
  FindScore(rank): Finds the score with the provided 0-based rank.
  FindScoreApproximate(rank): Finds a score >= the score of the provided 0-based
    rank, < the score of rank-1 (unless rank and rank-1 are tied, in which case
    it returns their mutual score).
  TotalRankedScores: The total number of scores in the Ranker.

  See __FindNodeIDs for more notes on structure.

  """

  def __init__(self, rootkey):
    """Pulls a ranker out of the datastore, given the key of the root node.

    Args:
      rootkey: The datastore key of the ranker.
    """
    # Get the root from the datastore:
    assert rootkey.kind() == "ranker"
    root = datastore.Get(rootkey)
    # Initialize some class variables:
    self.rootkey = rootkey
    self.score_range = root["score_range"]
    self.branching_factor = root["branching_factor"]
    # Sanity checking:
    assert len(self.score_range) > 1
    assert len(self.score_range) % 2 == 0
    for i in xrange(0, len(self.score_range), 2):
      assert self.score_range[i + 1] > self.score_range[i]
    assert self.branching_factor > 1

  @classmethod
  def Create(cls, score_range, branching_factor):
    """Constructs a new Ranker and returns it.

    Args:
      score_range: A list showing the range of valid scores, in the form:
        [most_significant_score_min, most_significant_score_max,
         less_significant_score_min, less_significant_score_max, ...]
        Ranges are [inclusive, exclusive)
      branching_factor: The branching factor of the tree.  The number of
        datastore Gets is Theta(1/log(branching_factor)), and the amount of data
        returned by each Get is Theta(branching_factor).

    Returns:
      A new Ranker.
    """
    # Put the root in the datastore:
    root = datastore.Entity("ranker")
    root["score_range"] = score_range
    root["branching_factor"] = branching_factor
    datastore.Put(root)
    myrank = Ranker(root.key())
    return myrank

  def __FindNodeIDs(self, score):
    """Finds the nodes along the path from the root to a certain score.

    Args:
      score: The score we're finding the path for.

    Returns:
      A sorted list of (node_id, child) tuples, indicating that node_id is the
      node id of a node on the path, and child is which child of that node is
      next.  Note that the lowest child node (which would be a leaf node) does
      not actually exist, since all its relevant information (number of times
      that score was inserted) is stored in its parent.

    Nodes are numbered row-by-row: the root is 0, its children are in the range
    [1, self.branching_factor + 1), its grandchildren are in the range
    [self.branching_factor + 1,
     self.branching_factor**2 + self.branching_factor + 1), etc.

    Score ranges are lists of the form: [min_0, max_0, min_1, max_1, ...]
    A node representing a score range will be divided up by the first index
    where max_i != min_i + 1 (score ranges are [inclusive, exclusive)).

    Child x (0-indexed) of a node [a,b) will get the range:
    [a+x*(b-a)/branching_factor, a+(x+1)*(b-a)/branching_factor);
    Thus not all nodes will have nonzero ranges.  Nodes with zero range will
    never be visited, but they and their descendants will be counted in the node
    numbering scheme, so row x still has self.branching_factor**x nodes.
    """
    nodes = []
    node = 0
    cur_range = list(self.score_range)
    # The current range of scores.  This will be narrowed as we move down the
    # tree; 'index' keeps track of the score type we're currently changing.
    for index in xrange(0, len(cur_range), 2):
      while cur_range[index + 1] - cur_range[index] > 1:
        # Subdivide cur_range[index]..cur_range[index + 1]
        which_child = self.__WhichChild(cur_range[index],
                                        cur_range[index + 1],
                                        score[index // 2],
                                        self.branching_factor)
        child = which_child[0]
        cur_range[index] = which_child[1][0]
        cur_range[index + 1] = which_child[1][1]
        assert 0 <= child < self.branching_factor
        nodes.append((node, child))
        node = self.__ChildNodeId(node, child)
    return nodes

  def __WhichChild(self, low, high, want, branching_factor):
    """Determines which child of the range [low, high) 'want' belongs to.

    Args:
      low: An int, the low end of the range.
      high: An int, the high end of the range.
      want: An int, the score we're trying to determine a child for.
      branching_factor: The branching factor of the tree being used.

    Returns:
      A tuple, (child, [child's score range]).  Note that in general a score
      has multiple sub-scores, written in order of decreasing significance; this
      function divides up a single sub-score.

    Raises:
      An AssertionError if things go horribly wrong.
    """
    assert low <= want < high
    # Need to find x such that (using integer division):
    #     x  *(high-low)/branching_factor <= want - low <
    #   (x+1)*(high-low)/branching_factor
    # Which is the least x such that (using integer division):
    #   want - low < (x+1)*(high-low)/branching_factor
    # Which is the ceiling of x such that (using floating point division):
    #   want - low + 1 == (x+1)*(high-low)/branching_factor
    #   x = -1 + math.ceil((want-low+1) * branching_factor / (high - low))
    # We get ceil by adding high - low - 1 to the numerator.
    x = -1 + (((want - low + 1) * branching_factor + high - low - 1) //
              (high - low))
    assert (x * (high - low) // branching_factor <=
            want - low < (x + 1) * (high - low) // branching_factor)
    return (x, self.__ChildScoreRange([low, high], x, branching_factor))

  def __ChildScoreRange(self, score_range, child, branching_factor):
    """Calculates the score_range for a node's child.

    Args:
      score_range: A score range [min0, max0, min1, max1, ...]
      child: Which child of the node with score range score_range we're
        calculating the score range of.
      branching_factor: The branching factor of the tree in question.

    Returns:
      A score range [min0', max0', min1', max1', ...] for that child.
    """
    for i in xrange(1, len(score_range), 2):
      if score_range[i] > score_range[i - 1] + 1:
        child_score_range = list(score_range)
        low, high = score_range[i - 1], score_range[i]
        child_score_range[i - 1], child_score_range[i] = (
            low + child * (high - low) // branching_factor,
            low + (child + 1) * (high - low) // branching_factor)
        return child_score_range
    raise AssertionError("Node with score range %s has no children." %
                         score_range)

  def __ChildNodeId(self, node_id, child):
    """Calculates the node id for a known node id's child.

    Args:
      node_id: The parent node's node_id
      child: Which child of the parent node we're finding the id for

    Returns:
      The node_id for the child'th child of node_id.
    """
    return node_id * self.branching_factor + 1 + child

  def __GetMultipleNodes(self, node_ids):
    """Gets multiple nodes from the datastore.

    Args:
      node_ids: A list of node ids we want to get.

    Returns:
      A dict of the nodes that were found, indexed by the node ids found
      in node_ids.
    """
    if len(node_ids) == 0:
      return []
    node_ids = set(node_ids)
    keys = [self.__KeyFromNodeId(node_id) for node_id in node_ids]
    nodes = datastore.Get(keys)
    return dict((node_id, node) for (node_id, node) in zip(node_ids, nodes)
                if node)

  # Although, this method is currently not needed, we'll keep this
  # since we might need it and some point and it's an interesting
  # relationship
  def __ParentNode(self, node_id):
    """Returns the node id of the parameter node id's parent.  Returns None if
    the parameter is 0."""
    if node_id == 0:
      return None
    return (node_id - 1) // self.branching_factor

  def __KeyFromNodeId(self, node_id):
    """Creates a (named) key for the node with a given id.

    The key will have the ranker as a parent element to guarantee
    uniqueness (in the presence of multiple rankers) and to put all
    nodes in a single entity group.

    Args:
      node_id: The node's id as an integer.

    Returns:
      A (named) key for the node with the id 'node_id'.
    """
    name = "node_%x" % node_id
    return datastore_types.Key.from_path("ranker_node", name,
                                         parent=self.rootkey)

  def __KeyForScore(self, name):
    """Returns a (named) key for a ranker_score entity.

    Args:
      name: Name of the score to create a key for.

    Returns:
      A (named) key for the entity storing the score of 'name'.
    """
    return datastore_types.Key.from_path("ranker_score", name,
                                         parent=self.rootkey)

  def __Increment(self, nodes_with_children, score_entities,
                  score_entities_to_delete):
    """Changes child counts for given nodes.

    This method will create nodes as needed.

    Args:
      nodes_with_children: A dict of (node_key, child) tuples to deltas
      score_entities: Additional score entities to persist as part of
        this transaction
    Returns:
      None
    """
    keys = list(set(key for ((key, _), delta) in nodes_with_children.iteritems()
                    if delta != 0))
    if not keys:
      return  # Nothing to do
    nodes = datastore.Get(keys)

    node_dict = {}
    for (key, node) in zip(keys, nodes):
      if not node:
        node = datastore.Entity("ranker_node", parent=self.rootkey,
                                name=key.name())
        node["child_counts"] = [0] * self.branching_factor
      node_dict[key] = node
    for ((key, child), amount) in nodes_with_children.iteritems():
      if amount != 0:
        node = node_dict[key]
        node["child_counts"][child] += amount
        assert node["child_counts"][child] >= 0
    datastore.Put(node_dict.values() + score_entities)
    if score_entities_to_delete:
      datastore.Delete(score_entities_to_delete)

  def SetScore(self, name, score):
    """Sets a single score.

    This is equivalent to calling 'SetScores({name: score})'

    Args:
      name: the name of the score as a string
      score: the score to set name to
    """
    return self.SetScores({name: score})

  @transactional
  def SetScores(self, scores):
    """Changes multiple scores atomically.

    Sets the scores of the named entities in scores to new values. For
    named entities that have not been registered with a score before,
    a new score is created. For named entities that already had a score,
    the score is changed to reflect the new score. If a score is None,
    the named entity's score will be removed from the ranker.

    Args:
      scores: A dict mapping entity names (strings) to scores (integer lists)
    """
    score_deltas, score_ents, score_ents_del = self.__ComputeScoreDeltas(scores)
    node_ids_to_deltas = self.__ComputeNodeModifications(score_deltas)
    self.__Increment(node_ids_to_deltas, score_ents, score_ents_del)

  def __ComputeScoreDeltas(self, scores):
    """Compute which scores have to be incremented and decremented.

    Args:
      scores: A dict mapping entity names to scores

    Returns:
      A tuple (score_deltas, score_entities, score_entities_to_delete).

      'score_deltas' is a dict, mapping scores (represented as tuples)
      to integers. 'score_deltas[s]' represents how many times the
      score 's' has to be incremented (or decremented).

      'score_entities' is a list of 'ranker_score' entities that have
      to be updated in the same transaction as modifying the ranker
      nodes. The entities already contain the updated score.

      Similarly, 'score_entities_to_delete' is a list of entities that
      have to be deleted in the same transaction as modifying the ranker
      nodes.
    """
    score_keys = [self.__KeyForScore(score) for score in scores]
    old_scores = {}
    for old_score in datastore.Get(score_keys):
      if old_score:
        old_scores[old_score.key().name()] = old_score
    score_deltas = {}
    # Score entities to update
    score_ents = []
    score_ents_del = []
    for score_name, score_value in scores.iteritems():
      if score_name in old_scores:
        score_ent = old_scores[score_name]
        if score_ent["value"] == score_value:
          continue  # No change in score => nothing to do
        old_score_key = tuple(score_ent["value"])
        score_deltas.setdefault(old_score_key, 0)
        score_deltas[old_score_key] -= 1
      else:
        score_ent = datastore.Entity("ranker_score", parent=self.rootkey,
                                     name=score_name)
      if score_value:
        score_key = tuple(score_value)
        score_deltas.setdefault(score_key, 0)
        score_deltas[score_key] += 1
        score_ent["value"] = score_value
        score_ents.append(score_ent)
      else:
        # Do we have to delete an old score entity?
        if score_name in old_scores:
          score_ents_del.append(old_scores[score_name])

    return (score_deltas, score_ents, score_ents_del)

  def __ComputeNodeModifications(self, score_deltas):
    """Computes modifications to ranker nodes.

    Given score deltas, computes which nodes need to be modified and by
    how much their child count has to be incremented / decremented.

    Args:
      score_deltas: A dict of scores to integers, as returned by
        _ComputeScoreDeltas.

    Returns:
      A dict of nodes (represented as node_key, child tuples) to integers.
      'result[(node_key, i)]' represents the amount that needs to be added to
      the i-th child of node node_key.
    """
    nodes_to_deltas = {}
    for score, delta in score_deltas.iteritems():
      for (node_id, child) in self.__FindNodeIDs(score):
        node = (self.__KeyFromNodeId(node_id), child)
        nodes_to_deltas[node] = nodes_to_deltas.get(node, 0) + delta
    return nodes_to_deltas

  def __FindRank(self, node_ids_with_children, nodes):
    """Utility function.  Finds the rank of a score.

    Args:
      node_ids_with_children: A list of node ids down to that score,
        paired with which child links to follow.
      nodes: A dict mapping node id to node entity.

    Returns:
      The score's rank.
    """
    tot = 0  # Counts the number of higher scores.
    for (node_id, child) in node_ids_with_children:
      if node_id in nodes:
        node = nodes[node_id]
        for i in xrange(child + 1, self.branching_factor):
          tot += node["child_counts"][i]
      else:
        # If the node isn't in the dict, the node simply doesn't exist.  We
        # are probably finding the rank for a score that doesn't appear in the
        # ranker, but that's perfectly fine.
        pass
    return tot

  def FindRank(self, score):
    """Finds the 0-based rank of a particular score; more precisely, returns the
    number of strictly higher scores stored.

    Args:
      score: The score whose rank we wish to find.

    Returns:
      The number of tracked scores that are higher.  Does not check whether
      anyone actually has the requested score.
    """
    return self.FindRanks([score])[0]

  def FindRanks(self, scores):
    """Finds the 0-based ranks of a number of particular scores.
    Like FindRank, but more efficient for multiple scores.

    Args:
      scores: A list of scores.

    Returns:
      A list of ranks.
    """
    for score in scores:
      assert len(score) * 2 == len(self.score_range)
    # Find the nodes we'll need to query to find information about these scores:
    node_ids_with_children_list = [self.__FindNodeIDs(score)
                                   for score in scores]
    node_ids = []
    for node_ids_with_children in node_ids_with_children_list:
      node_ids += [node_id for (node_id, _) in node_ids_with_children]
    # Query the needed nodes:
    nodes_dict = self.__GetMultipleNodes(node_ids)
    # Call __FindRank, which does the math, for each score:
    return [self.__FindRank(node_ids_with_children, nodes_dict) for
            node_ids_with_children in node_ids_with_children_list]

  def __FindScore(self, node_id, rank, score_range, approximate):
    """To be run in a transaction.  Finds the score ranked 'rank' in the subtree
    defined by node 'nodekey.'

    Args:
      node_id: The id of the node whose subtree we wish to find the
        score of rank 'rank' in.
      rank: The rank (within this subtree) of the score we wish to find.
      score_range: The score range for this particular node, as a list.
        Derivable from the node's node_id, but included for convenience.
      approximate: Do we have to return an approximate result, or an exact one?
        See the docstrings for FindScore and FindScoreApproximate.

    Returns:
      A tuple, (score, rank_of_tie), indicating the score's rank within
      node_id's subtree.  The way it indicates rank is defined in the dosctrings
      of FindScore and FindScoreApproximate, depending on the value of
      'approximate'.
    """
    # If we're approximating and thus allowed to do so, early-out if we just
    # need to return the highest available score.
    if approximate and rank == 0:
      return ([score - 1 for score in score_range[1::2]], 0)
    # Find the current node.
    node = datastore.Get(self.__KeyFromNodeId(node_id))
    child_counts = node["child_counts"]
    initial_rank = rank
    for i in xrange(self.branching_factor - 1, -1, -1):
      # If this child has enough scores that rank 'rank' is in
      # there, recurse.
      if rank - child_counts[i] < 0:
        child_score_range = self.__ChildScoreRange(score_range, i,
                                                   self.branching_factor)
        if self.__IsSingletonRange(child_score_range):
          # Base case; child_score_range refers to a single score. We don't
          # store leaf nodes so we can return right here.
          return (child_score_range[0::2], initial_rank - rank)
        # Not a base case.  Keep descending into children.
        ans = self.__FindScore(self.__ChildNodeId(node_id, i), rank,
                               child_score_range,
                               approximate)
        # Note the 'initial_rank - rank': we've asked the child for a score of
        # some rank among *its* children, so we have to add back in the scores
        # discarded on the way to that child.
        return (ans[0], ans[1] + (initial_rank - rank))
      else:
        rank -= child_counts[i]
    return None

  def __IsSingletonRange(self, scorerange):
    """Returns whether a range contains exactly one score."""
    return [score + 1 for score in scorerange[0::2]] == scorerange[1::2]

  @transactional
  def FindScore(self, rank):
    """Finds the score ranked at 'rank'.

    Args:
      rank: The rank of the score we wish to find.

    Returns:
      A tuple, (score, rank_of_tie).  'score' is the score ranked at 'rank',
      'rank_of_tie' is the rank of that score (which may be different from
      'rank' in the case of ties).
      e.g. if there are two scores tied at 5th and rank == 6, returns
      (score, 5).
    """
    return self.__FindScore(0, rank, self.score_range, False)

  @transactional
  def FindScoreApproximate(self, rank):
    """Finds a score that >= the score ranked at 'rank'.

    This method could be preferred to FindScore because it is more efficient.
    For example, if the objective is to find the top 50 scores of rank X or
    less, and those scores are stored in entities called scoreboard_row:
      score, rank = myrank.FindScoreApproximate(X)
      query = datastore.Query('scoreboard_row')
      query['score <='] = score
      result = query.Get(50 + X - rank)[X-rank:])  # Takes care of ties.

    Args:
      rank: The rank of the score we wish to find.

    Returns:
      A tuple, (score, rank_of_tie).
      If there is a tie at rank 'rank-1':
        rank's score <= score < rank-1's score, rank_of_tie == rank
      else:
        score == rank's score, rank_of_tie == the tied rank of everyone
        in the tie.
        e.g. if two scores are tied at 5th and rank == 6, returns (score, 5).
    """
    return self.__FindScore(0, rank, self.score_range, True)

  def TotalRankedScores(self):
    """Returns the total number of ranked scores.

    Returns:
      The total number of ranked scores.
    """
    root = datastore.Get([self.__KeyFromNodeId(0)])[0]
    if root:
      return sum(root["child_counts"])
    else:
      # Ranker doesn't have any ranked scores, yet
      return 0
