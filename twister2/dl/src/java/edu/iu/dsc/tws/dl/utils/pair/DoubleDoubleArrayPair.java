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
package edu.iu.dsc.tws.dl.utils.pair;


public class DoubleDoubleArrayPair implements PrimitiveArrayPair {
  private double t0;
  private double[] t1;

  public DoubleDoubleArrayPair() {
    //For Kryo
  }

  public DoubleDoubleArrayPair(double t0, double[] t1) {
    this.t0 = t0;
    this.t1 = t1;
  }

  public double getValue0() {
    return t0;
  }

  public double[] getValue1() {
    return t1;
  }

  public void setValue0(double value) {
    this.t0 = value;
  }

  public void setValue1(double[] value) {
    this.t1 = value;
  }
}
