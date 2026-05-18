import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Tensors as primitives',
    description: (
      <>
        Shapes are part of the type system. <code>f32[3, N]</code> is just a
        type. Element-wise <code>.+</code>, <code>.*</code>, and reductions
        like <code>+/</code> compose with the rest of the language without a
        ndarray library or runtime indirection.
      </>
    ),
  },
  {
    title: 'No LLVM dependency',
    description: (
      <>
        The compiler is pure Go. It includes its own x86-64 encoder, register
        allocator, and ELF object writer; only the system <code>ld</code> is
        invoked at the end. <code>go build ./cmd/esquec</code> and you are
        done.
      </>
    ),
  },
  {
    title: 'Pure-expression style',
    description: (
      <>
        Loops are spelled <code>tabulate</code>, <code>scan</code>, and{' '}
        <code>iterate_until</code>. There are no <code>for</code> /{' '}
        <code>while</code> keywords. Programs read top-down as data
        transformations, but compile to tight SIMD x86-64.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
