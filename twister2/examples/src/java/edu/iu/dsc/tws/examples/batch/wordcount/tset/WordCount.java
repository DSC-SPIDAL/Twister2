//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.examples.batch.wordcount.tset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.Twister2Worker;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.BaseSourceFunc;
import edu.iu.dsc.tws.examples.utils.RandomString;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.KeyedReduceTLink;
import edu.iu.dsc.tws.tset.sets.batch.KeyedTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

/**
 * A simple word count where we generate words in-memory
 */
public class WordCount implements Twister2Worker, Serializable {
  private static final Logger LOG = Logger.getLogger(WordCount.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);

    int sourcePar = 4;
    Config config = env.getConfig();

    // create a source with fixed number of random words
    SourceTSet<String> source = env.createSource(
        new WordGenerator((int) config.get("NO_OF_SAMPLE_WORDS"), (int) config.get("MAX_CHARS")),
        sourcePar).setName("source");
    // map the words to a tuple, with <word, 1>, 1 is the count
    KeyedTSet<String, Integer> groupedWords = source.mapToTuple(w -> new Tuple<>(w, 1));
    // reduce using the sim operation
    KeyedReduceTLink<String, Integer> keyedReduce = groupedWords.keyedReduce(Integer::sum);
    // print the counts
    keyedReduce.forEach(c -> LOG.info(c.toString()));
  }

  /**
   * A simple source, that generates words
   */
  class WordGenerator extends BaseSourceFunc<String> {
    private Iterator<String> iter;
    // number of words to generate by each source
    private int count;
    // max characters in each word
    private int maxChars;

    WordGenerator(int count, int maxChars) {
      this.count = count;
      this.maxChars = maxChars;
    }

    @Override
    public boolean hasNext() {
      return iter.hasNext();
    }

    @Override
    public String next() {
      return iter.next();
    }

    @Override
    public void prepare(TSetContext ctx) {
      super.prepare(ctx);
      Random random = new Random();
      RandomString randomString = new RandomString(maxChars, random, RandomString.ALPHANUM);
      List<String> wordsList = new ArrayList<>();
      for (int i = 0; i < count; i++) {
        wordsList.add(randomString.nextRandomSizeString());
      }
      // create a word list and use the iterator
      this.iter = wordsList.iterator();
    }
  }

  /**
   * We submit the job in the main method
   * @param args not using args for this job
   */
  public static void main(String[] args) {
    // build JobConfig, these are the parameters of the job
    JobConfig jobConfig = new JobConfig();
    jobConfig.put("NO_OF_SAMPLE_WORDS", 100);
    jobConfig.put("MAX_CHARS", 5);

    Twister2Job.Twister2JobBuilder jobBuilder = Twister2Job.newBuilder();
    jobBuilder.setJobName("tset-simple-wordcount");
    jobBuilder.setWorkerClass(WordCount.class);
    // we use 2 processes, each with 512mb memory and 1 CPU assigned
    jobBuilder.addComputeResource(1, 512, 4);
    jobBuilder.setConfig(jobConfig);

    // now submit the job
    Twister2Submitter.submitJob(jobBuilder.build(), ResourceAllocator.getDefaultConfig());
  }
}
