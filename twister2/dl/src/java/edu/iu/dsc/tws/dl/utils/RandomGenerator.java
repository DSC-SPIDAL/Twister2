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
package edu.iu.dsc.tws.dl.utils;


import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * A mersenne twister based fake random number generator.
 * Please refer https://en.wikipedia.org/wiki/Mersenne_Twister.
 * Note that it has its own state so it is not thread safe.
 * So you should use RandomGenerator.RNG to get a thread local instance to use.
 * That's thread-safe.
 */
@SuppressWarnings({"membername", "noclone", "superclone"})
public class RandomGenerator {
  private int MERSENNE_STATE_N = 624;
  private int MERSENNE_STATE_M = 397;
  private long MARTRX_A = 0x9908b0dfL;
  private long UMASK = 0x80000000L;
  /* most significant w-r bits */
  private long LMASK = 0x7fffffffL;
  /* least significant r bits */
  private String randomFileOS = "/dev/urandom";

  private long[] state = new long[MERSENNE_STATE_N];
  private long seed = 0;
  private int next = 0;
  private int left = 1;
  private double normalX = 0;
  private double normalY = 0;
  private double normalRho = 0;
  private boolean normalIsValid = false;

  public static ThreadLocal<RandomGenerator> generators = new ThreadLocal<>();

  public static RandomGenerator RNG() {
    if (generators.get() == null) {
      generators.set(new RandomGenerator());
    }
    return generators.get();
  }

  public RandomGenerator() {
    setSeed(randomSeed());
  }

  public RandomGenerator(long seed) {
    this.seed = seed;
  }

  @Override
  protected Object clone() throws CloneNotSupportedException {
    RandomGenerator result = new RandomGenerator();
    result.copy(this);
    return result;
  }

  public RandomGenerator copy(RandomGenerator from) {
    this.state = from.state.clone();
    this.seed = from.seed;
    this.next = from.next;
    this.normalX = from.normalX;
    this.normalY = from.normalY;
    this.normalRho = from.normalRho;
    this.normalIsValid = from.normalIsValid;
    return this;
  }

  private long randomSeed() {
    try {
      if (Files.exists(Paths.get(randomFileOS))) {
        FileInputStream fis = new FileInputStream(randomFileOS);
        BufferedInputStream bis = new BufferedInputStream(fis);
        byte[] buffer = new byte[8];
        bis.read(buffer, 0, 8);
        long randomNumber = ByteBuffer.wrap(buffer).getLong();
        bis.close();
        fis.close();
        return randomNumber;
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      //TODO: log error
    } catch (IOException e) {
      e.printStackTrace();
    }
    return System.nanoTime();
  }

  private long twist(long u, long v) {
    long a = ((u & UMASK) | (v & LMASK)) >> 1;
    long b = 0L;

    if ((v & 0x00000001L) != 0) {
      b = MARTRX_A;
    }

    return a ^ b;
  }

  RandomGenerator reset() {
    int i = 0;
    while (i < MERSENNE_STATE_N) {
      this.state[i] = 0L;
      i += 1;
    }

    this.seed = 0;
    this.next = 0;
    this.normalX = 0;
    this.normalY = 0;
    this.normalRho = 0;
    this.normalIsValid = false;
    return this;
  }

  RandomGenerator setSeed(long seedNum) {
    this.reset();
    this.seed = seedNum;
    this.state[0] = this.seed & 0xffffffffL;

    int i = 1;
    while (i < MERSENNE_STATE_N) {
      this.state[i] = 1812433253L * (this.state[i - 1] ^ (this.state[i - 1] >> 30)) + i;

      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
      /* In the previous versions, mSBs of the seed affect   */
      /* only mSBs of the array state[].                        */
      /* 2002/01/09 modified by makoto matsumoto           x  */
      this.state[i] = this.state[i] & 0xffffffffL; /* for >32 bit machines */
      i += 1;
    }
    this.left = 1;
    return this;
  }

  long getSeed() {
    return this.seed;
  }

  private RandomGenerator nextState() {
    int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1;
    int k = 0;

    this.left = MERSENNE_STATE_N;
    this.next = 0;

    while (j > 1) {
      j -= 1;
      this.state[k] = this.state[MERSENNE_STATE_M + k] ^ twist(this.state[k], this.state[k + 1]);
      k += 1;
    }

    j = MERSENNE_STATE_M;
    while (j > 1) {
      j -= 1;
      this.state[k] = this.state[MERSENNE_STATE_M - MERSENNE_STATE_N + k] ^ twist(this.state[k],
          this.state[k + 1]);
      k += 1;
    }

    this.state[k] = this.state[MERSENNE_STATE_M - MERSENNE_STATE_N + k] ^ twist(this.state[k],
        this.state[0]);
    return this;
  }

  /**
   * Generates a random number on [0,0xffffffff]-interval
   */
  public long random() {
    long y = 0;

    this.left = this.left - 1;
    if (this.left == 0) {
      this.nextState();
    }

    y = this.state[0 + this.next];
    this.next = this.next + 1;

    /* Tempering */
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680L;
    y ^= (y << 15) & 0xefc60000L;
    y ^= y >> 18;
    return y;
  }

  /**
   * Generates a random number on [0, 1)-real-interval
   */
  private double basicUniform() {
    return this.random() * (1.0 / 4294967296.0);
  }

  /**
   * Generates a random number on [a, b)-real-interval uniformly
   */
  public double uniform(double a, double b) {
    return this.basicUniform() * (b - a) + a;
  }

  public double normal(double mean, double stdv) {
    if (stdv <= 0) {
      throw new IllegalStateException("standard deviation must be strictly positive");
    }

    /* This is known as the Box-Muller method */
    if (!this.normalIsValid) {
      this.normalX = this.basicUniform();
      this.normalY = this.basicUniform();
      this.normalRho = Math.sqrt(-2 * Math.log(1.0 - this.normalY));
      this.normalIsValid = true;
    } else {
      this.normalIsValid = false;
    }

    if (this.normalIsValid) {
      return this.normalRho * Math.cos(2 * Math.PI * this.normalX) * stdv + mean;
    } else {
      return this.normalRho * Math.sin(2 * Math.PI * this.normalX) * stdv + mean;
    }
  }

  double exponential(double lambda) {
    return -1 / lambda * Math.log(1 - this.basicUniform());
  }

  double cauchy(double median, double sigma) {
    return median + sigma * Math.tan(Math.PI * (this.basicUniform() - 0.5));
  }

  double logNormal(double mean, double stdv) {
    double zm = mean * mean;
    double zs = stdv * stdv;
    if (stdv <= 0) {
      throw new IllegalStateException("standard deviation must be strictly positive");
    }
    return Math.exp(normal(Math.log(zm / Math.sqrt(zs + zm)), Math.sqrt(Math.log(zs / zm + 1))));
  }

  int geometric(double p) {
    if (!(p >= 0 && p <= 1)) {
      throw new IllegalStateException("must be >= 0 and <= 1");
    }
    return (int) ((Math.log(1 - this.basicUniform()) / Math.log(p)) + 1);
  }

  public boolean bernoulli(double p) {
    if (!(p >= 0 && p <= 1)) {
      throw new IllegalStateException("must be >= 0 and <= 1");
    }
    return this.basicUniform() <= p;
  }
}
